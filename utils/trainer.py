from models import DQN

import os
import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn.functional as F
from tqdm import tqdm

import torchutils

def parse_args():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training configuration file.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=[
            "frozenlake-v1",
            "cartpole-v1",
            "acrobot-v1",
            "taxi-v4",
            "mountaincar-v0",
            "cliffwalking-v1",
            "pendulum-v1",
            "blackjack-v1",
            "lunarlander-v3",
        ],
        default="frozenlake-v1",
        help="Environment to train on (default: frozenlake-v1).",
    )
    parser.add_argument(
        "--map_name",
        type=str,
        choices=["4x4"],
        default="4x4",
        help="Environment map to train on (default: 4x4).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100000,
        help="Total environment steps to train for (default: 100000).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.5e-4,
        help="Optimizer learning rate (default: 0.00025).",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="Replay buffer capacity in transitions (default: 10000).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size sampled from the replay buffer (default: 64).",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=1000,
        help="Environment steps before gradient updates begin (default: 1000).",
    )
    parser.add_argument(
        "--train_frequency",
        type=int,
        default=1,
        help="Environment steps between training updates (default: 1).",
    )
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=1,
        help="Gradient updates per training step (default: 1).",
    )
    parser.add_argument(
        "--target_update_frequency",
        type=int,
        default=1000,
        help="Environment steps between target-network updates (default: 1000).",
    )
    parser.add_argument(
        "--exploration_rate",
        type=float,
        default=0.2,
        help="Percentage of training steps over which epsilon is annealed (default: 0.2).",
    )
    parser.add_argument(
        "--exploration_initial_eps",
        type=float,
        default=1.0,
        help="Initial epsilon for epsilon-greedy exploration (default: 1.0).",
    )
    parser.add_argument(
        "--exploration_final_eps",
        type=float,
        default=0.05,
        help="Final epsilon for epsilon-greedy exploration (default: 0.05).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Number of layers in the network (default: 2).",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="Number of channels per layer (default: 256).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to write the final checkpoint to (default: ./checkpoints).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--action_values",
        type=int,
        default=11,
        help="Number of discrete action values for continuous-action environments.",
    )
    args = parser.parse_args()    
    return args

def main():
    args = parse_args()

    torchutils.set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    action_values = None

    if args.environment == "frozenlake-v1":
        env = gym.make("FrozenLake-v1", map_name=args.map_name, is_slippery=False)
        state_dim = env.observation_space.n
        action_dim = env.action_space.n
    elif args.environment == "cartpole-v1":
        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    elif args.environment == "acrobot-v1":
        env = gym.make("Acrobot-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    elif args.environment == "taxi-v4":
        env = gym.make("Taxi-v4")
        state_dim = env.observation_space.n
        action_dim = env.action_space.n
    elif args.environment == "mountaincar-v0":
        env = gym.make("MountainCar-v0")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    elif args.environment == "cliffwalking-v1":
        env = gym.make("CliffWalking-v1", max_episode_steps=1000)
        state_dim = env.observation_space.n
        action_dim = env.action_space.n
    elif args.environment == "pendulum-v1":
        env = gym.make("Pendulum-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = args.action_values
        action_min = float(env.action_space.low[0])
        action_max = float(env.action_space.high[0])
        action_values = [
            action_min + index * (action_max - action_min) / (action_dim - 1)
            for index in range(action_dim)
        ]
    elif args.environment == "blackjack-v1":
        env = gym.make("Blackjack-v1")
        state_dim = 3
        action_dim = env.action_space.n
    elif args.environment == "lunarlander-v3":
        env = gym.make("LunarLander-v3")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

    env.action_space.seed(args.random_seed)
    state, _ = env.reset(seed=args.random_seed)

    def encode_state(observation):
        if args.environment in ("frozenlake-v1", "taxi-v4", "cliffwalking-v1"):
            return F.one_hot(
                torch.as_tensor(observation, dtype=torch.long, device=device),
                num_classes=state_dim,
            ).float()
        if args.environment == "blackjack-v1":
            player_sum, dealer_card, usable_ace = observation
            return torch.tensor(
                [player_sum / 31, dealer_card / 10, float(usable_ace)],
                dtype=torch.float32,
                device=device,
            )
        return torch.as_tensor(observation, dtype=torch.float32, device=device)

    model = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        depth=args.depth,
        W=args.W,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    target_model = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        depth=args.depth,
        W=args.W,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    replay_buffer = deque(maxlen=args.buffer_size)
    episode_return = 0.0
    episode_returns = deque(maxlen=100)
    episode_count = 0
    exploration_steps = max(
        int(args.n_steps * args.exploration_rate),
        1,
    )

    pbar = tqdm(range(args.n_steps), desc="Training", unit="step")

    for step in pbar:
        progress = min(step / exploration_steps, 1.0)
        epsilon = args.exploration_initial_eps + progress * (
            args.exploration_final_eps - args.exploration_initial_eps
        )

        if random.random() < epsilon:
            if args.environment == "pendulum-v1":
                action = random.randrange(action_dim)
            else:
                action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(model(encode_state(state)).argmax().item())

        env_action = (
            [action_values[action]]
            if args.environment == "pendulum-v1"
            else action
        )
        next_state, reward, terminated, truncated, _ = env.step(env_action)
        replay_buffer.append(
            (state, action, reward, next_state, terminated)
        )
        episode_return += reward
        state = next_state

        if (
            step + 1 >= args.learning_starts
            and (step + 1) % args.train_frequency == 0
            and len(replay_buffer) >= args.batch_size
        ):
            for _ in range(args.gradient_steps):
                batch = random.sample(replay_buffer, args.batch_size)
                states, actions, rewards, next_states, terminations = zip(*batch)
                states_input = torch.stack([encode_state(s) for s in states])
                next_states_input = torch.stack(
                    [encode_state(s) for s in next_states]
                )
                actions_tensor = torch.tensor(
                    actions, dtype=torch.long, device=device
                ).unsqueeze(1)
                rewards_tensor = torch.tensor(
                    rewards, dtype=torch.float32, device=device
                )
                terminations_tensor = torch.tensor(
                    terminations, dtype=torch.float32, device=device
                )

                current_q_values = model(states_input).gather(
                    1, actions_tensor
                ).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_states_input).max(dim=1).values
                    target_q_values = rewards_tensor + args.gamma * (
                        1 - terminations_tensor
                    ) * next_q_values

                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        if (step + 1) % args.target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        if terminated or truncated:
            episode_count += 1
            episode_returns.append(episode_return)
            state, _ = env.reset()
            episode_return = 0.0

        average_return = (
            sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
        )
        pbar.set_postfix({
            "epsilon": f"{epsilon:.3f}",
            "reward": f"{average_return:.3f}",
        })
    env.close()

    checkpoint_dir = os.path.join(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    environment_name = args.environment
    map_name = None
    if args.environment == "frozenlake-v1":
        environment_name = f"{args.environment}-{args.map_name}"
        map_name = args.map_name
    checkpoint_file = os.path.join(
        checkpoint_dir, f"{environment_name}_{args.n_steps}.pth"
    )
    checkpoint = {
        "dqn": model.state_dict(),
        "environment": args.environment,
        "map_name": map_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "W": args.W,
        "depth": args.depth,
    }
    if action_values is not None:
        checkpoint["action_values"] = action_values
    torch.save(checkpoint, checkpoint_file)

if __name__ == "__main__":
    main()
