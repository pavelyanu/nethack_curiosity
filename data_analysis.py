import os
import re
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


class HyperparameterAnalysis:
    def __init__(
        self,
        log_dir,
        log_dir_suffix=".summary/0",
        metric_func=None,
        hyperparameters=None,
    ):
        self.root_dir = log_dir
        self.log_dir_suffix = log_dir_suffix
        self.metric_func = metric_func or self.default_metric_func
        self.hyperparameters = hyperparameters

    def parse_hyperparameters(self, log_dir_name):
        patterns = {
            "batch_size": r"b\.siz_(\d+)",
            "num_batches_per_epoch": r"n\.b\.p\.epo_(\d+)",
            "num_workers": r"n\.wor_(\d+)",
            "num_envs_per_worker": r"n\.e\.p\.wor_(\d+)",
            "env": r"env_([a-zA-Z0-9_]+)",
            "minigrid_room_size": r"m\.r\.siz_(\d+)",
            "minigrid_row_num": r"m\.r\.num_(\d+)",
            "train_for_env_steps": r"t\.f\.e\.ste_(\d+)",
            "recurrence": r"rec_(-?\d+)",
            "rollout": r"rol_(\d+)",
            "intrinsic_reward_weight": r"i\.r\.wei_([0-9\.]+)",
            "intrinsic_reward_module": r"i\.r\.mod_([a-zA-Z0-9_]+)",
        }

        if self.hyperparameters is not None:
            patterns = {
                key: pattern
                for key, pattern in patterns.items()
                if key in self.hyperparameters
            }

        hyperparameters = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, log_dir_name)
            if match:
                hyperparameters[key] = (
                    int(match.group(1)) if match.group(1).isdigit() else match.group(1)
                )

        return hyperparameters

    def extract_metrics(self, log_dir):
        all_metrics = []

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        scalars = ea.Tags()["scalars"]

        for scalar in scalars:
            events = ea.Scalars(scalar)
            for event in events:
                all_metrics.append(
                    {
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "metric": scalar,
                        "value": event.value,
                    }
                )
        df = pd.DataFrame(all_metrics)
        df = df.groupby(["step", "metric"], as_index=False).mean()
        df = df.pivot(index="step", columns="metric", values="value")

        # Interpolate missing values
        df = df.interpolate(method="linear", limit_direction="both")

        return df

    def get_subdirectories(self, directory):
        subdirs = [
            os.path.join(directory, subdir)
            for subdir in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, subdir))
        ]
        return subdirs

    def get_dataframes(self):
        dataframes = []
        for dir in self.get_subdirectories(self.root_dir):
            log_dir_name = os.path.basename(dir)
            hyperparameters = self.parse_hyperparameters(log_dir_name)
            log_dir = os.path.join(dir, self.log_dir_suffix)
            metrics = self.extract_metrics(log_dir)
            dataframes.append((hyperparameters, metrics))

        return dataframes

    def find_closest_step(self, metrics: pd.DataFrame, target_step):
        steps = metrics.index
        steps = np.array(steps)
        closest_step_index = np.argmin(np.abs(steps - target_step))
        return closest_step_index

    def default_metric_func(self, metrics):
        return metrics["reward/reward"].sum()

    def analyze_hyperparameter_importance(self, dataframes, include_lag=False):
        # Collect hyperparameters and the specified metric
        data = []
        for hyperparams, metrics in dataframes:
            if (
                "train_for_env_steps" in self.hyperparameters
                and hyperparams["train_for_env_steps"] > 5_000_000
            ):
                continue
            if "reward/reward" in metrics.columns:
                reward = self.metric_func(metrics)
                hyperparams["reward"] = reward
                if include_lag:
                    mean_lag = metrics["train/version_diff_avg"].mean()
                    hyperparams["mean_lag"] = mean_lag
                data.append(hyperparams)

        df = pd.DataFrame(data)

        # Remove hyperparameters with the same value across all experiments
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        df = df.drop(columns=constant_columns)

        # Prepare data for regression
        X = df.drop(columns=["reward"])
        y = df["reward"]

        # One-hot encode categorical variables (like 'env' and 'intrinsic_reward_module')
        X = pd.get_dummies(X)

        # Fit a single Decision Tree Regressor
        tree_regressor = DecisionTreeRegressor(random_state=42)
        tree_regressor.fit(X, y)

        return tree_regressor, X.columns, df

    def visualize_tree(self, tree_regressor, feature_names):
        plt.figure(figsize=(20, 10))
        plot_tree(tree_regressor, feature_names=feature_names, filled=True)
        plt.show()

    def print_top_results(self, data, n=10):
        top_results = data.sort_values("reward", ascending=False).head(n)
        print("Top results:")
        for index, row in top_results.iterrows():
            print(10 * "-")
            print("Result number:", index)
            print(row)

    def print_bottom_results(self, data, n=10):
        bottom_results = data.sort_values("reward", ascending=True).head(n)
        print("Bottom results:")
        for index, row in bottom_results.iterrows():
            print(10 * "-")
            print("Result number:", index)
            print(row)

    def print_sorted_results(self, data, by_columns, ascending=False):
        sorted_data = data.sort_values(by_columns, ascending=ascending)
        print(f"Results sorted by {by_columns}:")
        for index, row in sorted_data.iterrows():
            print(10 * "-")
            print("Result number:", index)
            print(row)

    @staticmethod
    def sum_reward(metrics):
        return metrics["reward/reward"].sum()

    @staticmethod
    def last_reward(metrics):
        return metrics["reward/reward"].dropna().iloc[-1]

    @staticmethod
    def reward_at_step(metrics, target_step):
        closest_step_index = np.argmin(np.abs(np.array(metrics.index) - target_step))
        return metrics.iloc[closest_step_index]["reward/reward"]

    @staticmethod
    def print_seperator():
        print(100 * "#")
        print(100 * "#")


# Example usage
if __name__ == "__main__":
    # root_dir = "/home/pavel/dev/thesis/project/minigrid_experiments_train_dir/minigrid_bs_bpe_grid_at_rnd_0.01/minigrid_"
    root_dir = (
        "/home/pavel/dev/thesis/project/minigrid_rol128_train_dir/rol128/keyS3R1rol128_"
    )

    # Choose the desired metric function
    # metric_function = HyperparameterAnalysis.reward_at_step
    metric_function = HyperparameterAnalysis.last_reward

    hyperparameters_to_parse = [
        "batch_size",
        "num_batches_per_epoch",
        "num_workers",
        "num_envs_per_worker",
        "env",
        "minigrid_room_size",
        "minigrid_row_num",
        # "train_for_env_steps",
        "recurrence",
        "rollout",
        "intrinsic_reward_weight",
        "intrinsic_reward_module",
    ]

    analysis = HyperparameterAnalysis(
        root_dir,
        # metric_func=lambda metrics: metric_function(metrics, 1_000_000),
        metric_func=metric_function,
        hyperparameters=hyperparameters_to_parse,
    )

    dataframes = analysis.get_dataframes()
    tree_regressor, feature_names, data = analysis.analyze_hyperparameter_importance(
        dataframes
    )

    print("Feature importances:")
    feature_importances = list(zip(feature_names, tree_regressor.feature_importances_))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    for name, importance in feature_importances:
        print(f"{name}: {importance}")

    analysis.visualize_tree(tree_regressor, feature_names)

    # Print top 10 results with the highest reward
    analysis.print_top_results(data, n=10)
    analysis.print_seperator()

    # Print top 10 results with the lowest reward
    analysis.print_bottom_results(data, n=10)
    analysis.print_seperator()

    # Print all results sorted by reward
    analysis.print_sorted_results(data, by_columns=["reward"], ascending=False)
    analysis.print_seperator()

    # Print all results sorted by batch_size, num_batches_per_epoch, num_workers, num_envs_per_worker
    analysis.print_sorted_results(
        data,
        by_columns=[
            "batch_size",
            "num_batches_per_epoch",
            "num_workers",
            "num_envs_per_worker",
        ],
    )
