
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from agents import *
from envs import *
from utils import *

def _get_env_shapes(env_id, env_type="atari"):
    if env_type == "atari":
        env = gym.make(env_id)
    else:
        raise NotImplementedError(f"Unsupported env_type: {env_type}")

    input_size = env.observation_space.shape
    output_size = env.action_space.n
    env.close()
    return input_size, output_size


def create_agent(config):
    env_id = config["EnvID"]
    env_type_str = config["EnvType"].lower()
    train_method = config["TrainMethod"].lower()

    sticky_action = config.getboolean("StickyAction")
    action_prob = float(config.get("ActionProb", 0.0))
    life_done = config.getboolean("LifeDone")

    use_cuda = config.getboolean("UseGPU")
    use_gae = config.getboolean("UseGAE")
    use_noisy_net = config.getboolean("UseNoisyNet")

    lam = float(config["Lambda"])
    num_worker = int(config["NumEnv"])
    num_step = int(config["NumStep"])

    ppo_eps = float(config["PPOEps"])
    epoch = int(config["Epoch"])
    mini_batch = int(config["MiniBatch"])
    learning_rate = float(config["LearningRate"])

    entropy_coef = float(config["Entropy"])
    gamma = float(config["Gamma"])
    eta = None
    if train_method in ["rffrnd"]:
        eta = float(config["ETA"])
    ext_coef = float(config["ExtCoef"])
    int_coef = float(config["IntCoef"])

    clip_grad_norm = float(config.get("ClipGradNorm", 0.5))
    update_proportion = float(config.get("UpdateProportion", 0.25))

    lengthscale = float(config.get("Lengthscale", 1.0))
    base_loss_weight = 1.0
    alpha = float(config.get("Alpha", 0.95))
    num_target = int(config.get("NumTarget", 1))
    batch_size = int(num_step * num_worker / mini_batch)

    input_size, output_size = _get_env_shapes(env_id, env_type=env_type_str)

    if env_type_str == "atari":
        env_cls = AtariEnvironment
    else:
        raise NotImplementedError(f"Unknown EnvType: {env_type_str}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_env = int(config["NumEnv"])
    entropy_coef = float(config["Entropy"])
    int_coef = float(config["IntCoef"])
    tags = [
        f"nenv{num_env}",
        f"ent{entropy_coef:g}",
        f"intc{int_coef:g}",
    ]
    if train_method == "rffrnd":
        tags.append(f"ls{lengthscale:g}")

    run_name = f"{env_type_str}_{env_id}_" + "_".join(tags) + f"_{timestamp}"
    log_dir = os.path.join("runs", train_method, run_name)

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TensorBoard Save] log_dir = {log_dir}")

    agent_map = {
        "RFFRND" : RFFRNDAgent,
        "rffrnd" : RFFRNDAgent
    }
    if train_method not in agent_map:
        raise ValueError(f"Unknown TrainMethod: {train_method}")

    AgentCls = agent_map[train_method]
    agent_kwargs = dict(
        input_size=input_size,
        output_size=output_size,
        num_env=num_worker,
        num_step=num_step,
        gamma=gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
    )
    if train_method in ["rffrnd", "rff_rnd", "rffrnd"]:
        output_dim = int(config["OutputDim"])
        agent_kwargs["output_dim"] = output_dim


    if train_method in ["rffrnd", "rnd", "rff_rnd", "rffrnd"]:
        agent_kwargs.update(
            dict(
                clip_grad_norm=clip_grad_norm,
                update_proportion=update_proportion,
                num_target=num_target,
                base_loss_weight=base_loss_weight,
                alpha=alpha,
                eta=eta,
            )
        )

    if train_method == "rffrnd":
        agent_kwargs["lengthscale"] = lengthscale

    agent = AgentCls(**agent_kwargs)
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))

    int_gamma = float(config.get("IntGamma", 0.99))
    pre_obs_norm_step = int(config.get("ObsNormStep", 0))

    extra = dict(
        sticky_action=sticky_action,
        action_prob=action_prob,
        life_done=life_done,
        input_size=input_size,
        output_size=output_size,
        env_id=env_id,
        gamma=gamma, 
        env_cls=env_cls,
        reward_rms=reward_rms,
        num_worker=num_worker,
        num_step=num_step,
        obs_rms=obs_rms,
        int_gamma=int_gamma,
        pre_obs_norm_step=pre_obs_norm_step,
        ext_coef=ext_coef,
        int_coef=int_coef,    
    )

    return agent, writer, log_dir, extra