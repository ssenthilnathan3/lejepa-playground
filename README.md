# An implementation try to build a LeJEPA playground

## Observations

### Observation - 1 (JEPA)
- For a simple world (GridWorld) which has a 10x10 grid, the agent seems to figure out that 
  z_pred = z_t
- The loss becomes 0 at 400th step
- This is not latent collapse
    - If loss became exactly 0 at step ~400, that is not real learning - it means the model found a shortcut and is *cheating the loss*.
    - Meaning the encoder produces a constant latent making the z_t1 produce the same latent eventually figuring out z_pred = z_t1 which collapses the entire model

This is solved by the SIGReg loss instead of MSE loss in LeJEPA. Continuous spaces instead of simpler environments

BallWorld fits perfectly here, I'm gonna try the same setup and train jepa with BallWorld which has a continous space as opposed to GridWorld which has a discrete one.

### Observation 2 - (JEPA Split)

- For BallWorld, eventhough z_pred was meaningful. There was no learning or understanding of latent space by the model
- By tweaking the parameters such as velocity, change of position over time (dt). I could make the model react in the environment
- But still the problem was linearity and identity/bias collapse. i.e z_obs eventually becomes equal to z_pred
- So, with some brainstorming, I split the encoder, dynamics even the training to focus more on learning the physical system.
- Because the dynamics MLP is still trying to infer velocity from position changes in the latent, which is incredibly hard. So, I split latent into *z_vel* and *z_pos*
- MLPs tend to overfit position and underfit velocity, because:
    - x, y are large-scale
    - vx, vy are small contributions
    - even scaled vx/vy get overshadowed
