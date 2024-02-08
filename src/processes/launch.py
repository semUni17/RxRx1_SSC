import os
import sys
import argparse

sys.path.append(os.getcwd())


def main():
    if args.net == "simclr":
        print("SimCLR")
        from src.methods.selfsupervised.simclr.train.train import Train
        from src.methods.selfsupervised.simclr.test.test import Test
    elif args.net == "classifier":
        print("Classifier")
        from src.methods.classifier.train.train import Train
        from src.methods.classifier.test.test import Test
    else:
        print("Error occurred in passing arguments")
        return

    if args.mode == "train":
        print("Train")
        training = Train(args.config_file)
        training.train()
    elif args.mode == "test":
        print("Test")
        testing = Test(args.config_file)
        testing.test()
    else:
        print("Error occurred in passing arguments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net", required=False, help="Network to train or test (simclr/classifier)")
    parser.add_argument("-m", "--mode", required=False, help="Mode (train/test)")
    parser.add_argument("-cfg", "--config_file", required=True, help="Path to .yml config file")
    args = parser.parse_args()
    main()
