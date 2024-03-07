import neat
import logging
from reporter import ResultsReporter
import os
import multiprocessing
from parallel import ParallelEvaluator
import numpy as np


class TrainerNEAT:
    def __init__(self, config, x_data, y_data, x_test):
        """
        Runs the NEAT algorithm and orchestrates the evaluation process.
        """
        self.config = config
        self.t = 1000  # number of generations
        self.restore_ckpt = True  # restore from the last checkpoint?
        self.ckpt_path = "./checkpoints/"  # "/egr/scratch/zamojcin/checkpoints/"
        self.ckpt_prefix = os.path.join(self.ckpt_path, "neat-ckpt-")
        self.p = None  # population instance
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test

    def run(self):
        # Create or restore the population, which is the top-level object for a NEAT run.
        if self.restore_ckpt:
            last_ckpt = self.get_last_ckpt()
            if last_ckpt >= 0:
                print(f"Restoring population from checkpoint {last_ckpt}...")
                self.p = neat.Checkpointer.restore_checkpoint(f'{self.ckpt_prefix}{last_ckpt}')
                self.p.generation += 1
                self.p.config.pop_size = self.config.pop_size
                # self.p.config = self.config
                # self.p.species.species_set_config.compatibility_threshold = self.config.species_set_config.compatibility_threshold
                # self.p.reproduction.stagnation.stagnation_config.max_stagnation = self.config.stagnation_config.max_stagnation
        if not self.p:
            print("Creating initial population...")
            self.p = neat.Population(self.config)

        # init checkpointer
        checkpointer = neat.Checkpointer(
           #  generation_interval=50,
            time_interval_seconds=3600,
            filename_prefix=self.ckpt_prefix
        )

        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(True))
        self.p.add_reporter(ResultsReporter(self._init_logger()))
        stats = neat.StatisticsReporter()
        self.p.add_reporter(stats)
        self.p.add_reporter(checkpointer)
        print(f"Init generation #: {self.p.generation}")
        print(f"Init population size #: {len(self.p.population)}")

        # Run
        print("Starting run...\n")
        winner = self.p.run(self.eval, self.t)
        # print(f"\nWinner fitness: {winner.fitness}")
        self.predict(winner, self.config, self.x_test)

    def eval(self, genomes, config):
        p_eval = ParallelEvaluator(num_workers=64, eval_function=self._eval_genome)
        p_eval.evaluate(genomes, config, (self.x_data, self.y_data))

    @staticmethod
    def _eval_genome(genome, config, data) -> float:
        x_data, y_data = data
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        for x, y in zip(x_data, y_data):
            output_layer = net.activate(x)
            y_pred = output_layer.index(max(output_layer))
            fitness += 1.0 if y == y_pred else 0.0
        fitness /= float(len(y_data))
        # print(f"\t{genome.key}: {fitness}")
        return fitness

    def predict(self, genome, config, x_data) -> None:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        preds = []
        for x in x_data:
            output_layer = net.activate(x)
            y = output_layer.index(max(output_layer))
            preds.append(y)
        preds = np.array(preds) # 1984 lines
        print(f"Prediction shape: {preds.shape}")
        np.savetxt('submission.txt', preds, fmt='%d')
        # print(f"Winner fitness: {winner_fit}")

    def get_last_ckpt(self) -> int:
        highest_idx = -1
        for f in os.listdir(self.ckpt_path):
            split = f.split("neat-ckpt-")
            if len(split) == 2:
                highest_idx = max(highest_idx, int(split[1]))
        return highest_idx

    @classmethod
    def _init_logger(cls):
        """
        Initializes the Trainer logger.
        Logs NEAT training results to logs/trainer.log and debug logs to console.
        """
        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter('%(message)s')

        # remove any existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        info_handler = logging.FileHandler('./logs/trainer.log')
        info_handler.setFormatter(log_format)
        info_handler.setLevel(logging.INFO)
        logger.addHandler(info_handler)
        return logger
