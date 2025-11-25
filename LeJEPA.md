# LeJEPA

## Core Idea

- Two encoders: Context Encoder and Target Encoder
- They produce the same type of latent (because I choose to reuse the split training method)
- They both process raw obs

You might be thinking *Ohh they are twins* but structurally they are same but functionally, one moves based on gradient and another moves slowly
