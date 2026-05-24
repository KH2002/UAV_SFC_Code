MLP

单encoder

encoder+cross_atten

encoder+agent_atten+cross_atten:
    MAPPO/output/mappo_seed42_20260510_223622:200wstep
    MAPPO/output/60wstep_encoder_agent-atten_cross-atten_data100:60wstep

    课程学习


**需要对比的强化学习算法**

    1、TD3


**需要验证的改进**
    1、集群MA，同时输出所有agent的动作
    2、