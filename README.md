# DiVAS.

## Overview
In this project, I am trying to create a software that optimizes my daily schedule. The background theory is based on neurobiology - No screen 1 hour before sleep, etc. The flow is that I'll probably download TinyLlama (coz of space) and use DPO to fine tune it for neurobiology. A sanity check would be to ask it so solve a basic physics questions and it SHOULD fail.

There are two key components - 
    1. Google Calendar Binding (done)
    2. vanilla, uninformed RL on current blank schedule (doing)
    3. DPO on the LLM (tbd)


The higher-level architecture would be like this

         +------------+       +------------+       +-------------+
         |  Day In    | ----> |  RL Model  | ----> |   Day Out   |
         | (Schedule) |       |   (PPO)    |       | (Optimized) |
         +------------+       +------------+       +-------------+
                                   ^
                                   |
                             +-------------+
                             |     LLM     |
                             |  (Advisor)  |
                             +-------------+




## License
MIT License. See `LICENSE` for details.


# To do:
 - PPO for basic schedule

# Project Updates:

### July 22, 2025

I got the actor and critic models working. But I have a feeling they are getting stuck in a local loss minimum so I added a "spiking" functionality. Essentially slightly modifies the parameters by multiplying in some noise and hopefully the model "escapes" the local minima and explores more.
