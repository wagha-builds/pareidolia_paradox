import argparse
import time
from .utils import set_seed, setup_logging
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    set_seed(42)
    setup_logging()
    
    logging.info(f"Loaded config from {args.config}")
    logging.info("Initializing model (stub)...")
    
    # Simulate a fast training run
    epochs = 5
    loss = 1.0
    for epoch in range(1, epochs + 1):
        loss *= 0.8
        logging.info(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - val_BA: {0.5 + (0.5 * (1-loss)):.4f}")
        time.sleep(0.1)
        
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
