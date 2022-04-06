""" Blackbox Attack [Project: visualsemantic]

This script performs "blackbox attack" on encoders by generating predicted voxels from synthesized images
and comparing the average activations with normal images. 

This script contains the following functions and classes:

    * RegressionLazy
    * wrap_betas

"""

from argparse import ArgumentParser
import os
import pickle
import numpy as np
import xarray as xr
from scipy import spatial
import scipy.stats as stats

import matplotlib.pyplot as plt
from bonner.models.alexnet_imagenet import alexnet_imagenet
from sklearn import preprocessing

from src.utils import listdir

import torch
import xarray as xr

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore.metrics.xarray_utils import map_target_to_source
from brainscore.metrics.regression import (
    linear_regression_efficient,
    pearsonr_correlation_efficient,
)
from bonner.brainscore.benchmarks.bonner2021_object2vec import (
    Bonner2021Object2VecBenchmark,
)
from model_tools.activations.hooks import GlobalMaxPool2d
from model_tools.brain_transformation.neural import LayerMappedModel, PreRunLayers
from model_tools.brain_transformation.temporal import TemporalIgnore


class RegressionLazy:
    def __init__(
        self,
        regression,
        correlation,
        stimulus_dim="presentation",
        stimulus_coord="image_id",
    ):
        self.regression = regression
        self.correlation = correlation
        self.stimulus_dim = stimulus_dim
        self.stimulus_coord = stimulus_coord

    def __call__(self, source, target):
        return self.apply(source, target)

    def apply(self, source, target):
        self.regression.fit(source, target)
        source_test = source.isel(
            {
                self.stimulus_dim: map_target_to_source(
                    source, target, self.stimulus_coord
                )
            }
        )
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target)
        aggregated_score = self.aggregate(score)
        aggregated_score.attrs["raw"] = score
        return aggregated_score

    def aggregate(self, scores):
        return scores.median(dim="neuroid")


def wrap_betas(
    benchmark: Bonner2021Object2VecBenchmark, model_assembly: NeuroidAssembly
):
    betas = {}
    for subject, metric in benchmark.metrics.items():
        betas[subject] = metric.regression._regression.betas
        betas[subject] = (
            xr.DataArray(
                data=betas[subject],
                dims=("neuroid_model", "neuroid_neural"),
            )
            .assign_coords(
                {
                    coord: ("neuroid_neural", values)
                    for coord, dims, values in walk_coords(
                        benchmark.assemblies[subject]
                    )
                    if len(dims) == 1 and dims[0] == "neuroid"
                }
            )
            .assign_coords(
                {
                    coord: ("neuroid_model", values)
                    for coord, dims, values in walk_coords(model_assembly)
                    if len(dims) == 1 and dims[0] == "neuroid"
                }
            )
        )
    return betas


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layers")
    args = parser.parse_args()
    layers = [args.layers]

    # Set dataset
    dataset = "bonner2021.object2vec"

    # Set ROIs
    rois = ["LOC", "PPA", "EVC"]
    betas = {roi: [] for roi in rois}

    # Extract betas
    for roi in rois:
        benchmark = Bonner2021Object2VecBenchmark(
            identifier=f"{dataset}.my-encoder",
            metric=RegressionLazy(
                regression=linear_regression_efficient(),
                correlation=pearsonr_correlation_efficient(device=torch.device("cpu")),
            ),
            ceiling_func=None,
            stimulus_background="texture",
            kwargs_loader={
                "subjects": None,
                "z_score": True,
                "average_reps": False,
                "check_integrity": False,
            },
            kwargs_filter={
                "rois": (roi,),
            },
        )

        for i_layer, layer in enumerate(layers):
            layer_model = LayerMappedModel(
                identifier=alexnet_imagenet.identifier,
                activations_model=alexnet_imagenet,
                region_layer_map={None: layer},
                visual_degrees=None,
            )
            layer_model = TemporalIgnore(layer_model)
            if i_layer == 0:
                layer_model = PreRunLayers(
                    model=alexnet_imagenet, layers=layers, forward=layer_model
                )
            model_assembly = benchmark._extract_features(layer_model)
            _ = benchmark._compute_score(model_assembly)
            betas[roi] = wrap_betas(benchmark, model_assembly)[0].values

    # Load input images
    conditions = ["face", "scene", "bodypart", "object"]
    condition_imgs = {c: [] for c in conditions}
    for condition in conditions:
        stim_path = "../../visualsemantic/stimuli/" + condition
        images = listdir(stim_path)
        condition_imgs[condition] = [img.split("/")[-1] for img in images]

    result_path = "../../visualsemantic/results/adversary"
    results = [item.split("/")[-1] for item in listdir(result_path) if ".png" in item]

    synthesized_outputs = [item for item in results if len(item.split("_")) > 1]
    content_inputs = list(
        set([item.split("_")[0] + ".png" for item in synthesized_outputs])
    )
    style_inputs = list(
        set([item.split("_")[1].split(".")[0] + ".png" for item in synthesized_outputs])
    )

    # Get activations for input and output images    
    content_activations = alexnet_imagenet(
        [os.path.join(result_path, ci) for ci in content_inputs], layers=layers, stimuli_identifier="content." + " ".join(layers)
    ).values
    
    style_activations = alexnet_imagenet(
        [os.path.join(result_path, si) for si in style_inputs], layers=layers, stimuli_identifier="style." + " ".join(layers)
    ).values
    
    output_activations = alexnet_imagenet(
        [os.path.join(result_path, o) for o in synthesized_outputs], layers=layers, stimuli_identifier="synthesized." + " ".join(layers)
    ).values

    # Attack

    for roi in rois:

        content_predicted = {}
        style_predicted = {}
        output_predicted = {}

        nImgsOutput = len(output_activations)
        nImgsContent = len(content_activations)
        nImgsStyle = len(style_activations)

        # normalize and get predicted activations
        data = np.vstack([v for v in list(content_activations.values())])
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        for i in range(nImgsContent):
            key = list(content_activations.keys())[i]
            content_activations[key] = data[i, :]
            content_predicted[key] = np.matmul(content_activations[key], betas[roi])

        data = np.vstack([v for v in list(style_activations.values())])
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        for i in range(nImgsStyle):
            key = list(style_activations.keys())[i]
            style_activations[key] = data[i, :]
            style_predicted[key] = np.matmul(style_activations[key], betas[roi])

        data = np.vstack([v for v in list(output_activations.values())])
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        for i in range(nImgsOutput):
            key = list(output_activations.keys())[i]
            output_activations[key] = data[i, :]
            output_predicted[key] = np.matmul(output_activations[key], betas[roi])

        # create rdms
        rdm_output = np.zeros((nImgsOutput, nImgsOutput))
        rdm_content = np.zeros((nImgsOutput, nImgsOutput))
        rdm_style = np.zeros((nImgsOutput, nImgsOutput))

        for i, c in enumerate(sorted(output_activations.keys())):
            for j, d in enumerate(sorted(output_activations.keys())):

                r, p = stats.pearsonr(output_predicted[c], output_predicted[d])
                # r = 1 - spatial.distance.cosine(style_activations[c], style_activations[d])
                rdm_output[i, j] = r

        for i, c in enumerate(sorted(output_activations.keys())):
            for j, d in enumerate(sorted(output_activations.keys())):
                file_c = c.split("_")[0] + ".png"
                file_d = d.split("_")[0] + ".png"
                r, p = stats.pearsonr(
                    content_predicted[file_c], content_predicted[file_d]
                )
                # r = 1 - spatial.distance.cosine(content_activations[c], content_activations[d])
                rdm_content[i, j] = r

        for i, c in enumerate(sorted(output_activations.keys())):
            for j, d in enumerate(sorted(output_activations.keys())):
                file_c = c.split("_")[1].split(".")[0] + ".png"
                file_d = d.split("_")[1].split(".")[0] + ".png"
                r, p = stats.pearsonr(style_predicted[file_c], style_predicted[file_d])
                # r = 1 - spatial.distance.cosine(style_activations[c], style_activations[d])
                rdm_style[i, j] = r

        triu_ind = np.triu_indices(nImgsOutput, k=1)
        r, p = stats.spearmanr(
            rdm_output[triu_ind], rdm_content[triu_ind], axis=None
        )  # compute rank correlation between each model and each roi rdm
        rsa[roi]["content"] = r
        r, p = stats.spearmanr(rdm_output[triu_ind], rdm_style[triu_ind], axis=None)
        rsa[roi]["style"] = r
        print(rsa)

    f = open("../../visualsemantic/results/rsa/" + " ".join(layers) + ".pkl", "wb")
    pickle.dump(rsa, f)
    f.close()
