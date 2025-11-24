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
