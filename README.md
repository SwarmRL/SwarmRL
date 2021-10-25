# ML meets Espresso

Repository to study the application of ML models to espresso simulations.

## RL colloids
My concept of the workflow with definitions

1. Define RL model
2. Define espresso simulator
3. Create environment containing both of the model and the simulator.
4. Run environment.train() to train the model using the simulator as a data generator.
5. Run environment.run_production() To run a production simulation with a trained model.

Parent classes exists but the environment and simulator need to be filled out.