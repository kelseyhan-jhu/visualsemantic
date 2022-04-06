""" Blackbox Attack [Project: visualsemantic]

This script performs "blackbox attack" on encoders by generating predicted voxels from synthesized images
and comparing the average activations with normal images. 

This script contains the following functions and classes:


"""

from argparse import ArgumentParser
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import listdir

from brainscore.metrics.regression import linear_regression
from brainio.fetch import get_stimulus_set
from model_tools.brain_transformation.neural import LayerMappedModel
from bonner.models.alexnet_imagenet import alexnet_imagenet
from bonner.brainscore.benchmarks.bonner2021_object2vec import load_assembly, extract_features


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layers", default="features.10")
    parser.add_argument("--subject", type=int, default=0)
    args = parser.parse_args()
    subject = args.subject
    layers = [args.layers]

    # Set dataset
    dataset = "bonner2021.object2vec"

    # Set ROIs
    rois = ['FFA', 'EBA', 'PPA', 'LOC', 'EVC']

    # getting an ActivationsModel (basically a normal Pytorch model wrapped in model_tools.activations.pytorch.PytorchWrapper)
    model = alexnet_imagenet()

    # wrapping that model in model_tools.brain_transformation.neural.LayerMappedModel to convert it into a brainscore.model_interface.BrainModel
    candidate_model = LayerMappedModel("alexnet"+layers[0], model, {None: layers[0]})

    # using the images with textured backgrounds
    stimulus_set = get_stimulus_set("bonner2021.object2vec-texture")

    # using an extract_features function defined by me instead of directly doing model(stimulus_set) because Object2Vec has some annoyances (need to average across images within a category since it's a block design)
    model_assembly = extract_features(candidate_model, stimulus_set)

    print(model_assembly.values.shape)
    # extract betas
    betas = {roi: [] for roi in rois}

    for roi in rois:
        print(roi)
        neural_assembly = load_assembly(
            subject,
            average_reps=False,
            z_score=True,
            check_integrity=False,
            kwargs_filter={
                "rois": (
                    roi,

                )
            }
        )
        regression = linear_regression(
            backend="pytorch",
            torch_kwargs={"device": "cpu"},
        )
        regression.fit(model_assembly, neural_assembly)
        # if using the Pytorch backend I wrote, the betas are currently stored in the `betas` attribute
        # I'll change it to follow the sklearn API later to be more consistent
        betas[roi] = regression._regression.betas

    ## Load input betas from Murty et al. images
    N_IMAGES = 10

    f = h5py.File('/home/chan21/projects/visualsemantic/stimuli/L2_data.mat', 'r')
    L2data = f.get('L2_data')
    print(L2data.keys())
    L2data_rois = (L2data.get('rois'))

    conditions = ["face", "place", "body", "object"]
    condition_idx = {"face": np.arange(26, 51),
                     "place": np.concatenate((np.arange(76, 101), np.arange(126, 151)), axis=None),
                     "body": np.concatenate((np.arange(1, 26), np.arange(101, 126)), axis=None),
                     "object": np.concatenate((np.arange(51, 76), np.arange(151, 186)), axis=None)}

    pooled_obs = np.array(L2data.get('pooled_obs'))
    pooled_pred = np.array(L2data.get('pooled_pred'))

    best_face = np.stack((pooled_pred[:, 0], pooled_pred[:, 1]), axis=1).mean(axis=1).argsort()[-N_IMAGES:][::-1]
    best_body = np.stack((pooled_pred[:, 2], pooled_pred[:, 3]), axis=1).mean(axis=1).argsort()[-N_IMAGES:][::-1]
    best_place = np.stack((pooled_pred[:, 4], pooled_pred[:, 5]), axis=1).mean(axis=1).argsort()[-N_IMAGES:][::-1]
    best_object = condition_idx["object"][:N_IMAGES]

    condition_best_idx = {"face": best_face,
                          "body": best_body,
                          "place": best_place,
                          "object": best_object}


    ## Get activations for encoder-predicted top-category images

    condition_imgs = {c: [] for c in conditions}
    stim_path = "/home/chan21/projects/visualsemantic/stimuli/stimuli"
    stim_images = listdir(stim_path)
    for condition in conditions:
        condition_imgs[condition] = [stim_images[idx] for idx in condition_best_idx[condition]]

    condition = 'face'
    face_activations = alexnet_imagenet()(
        [ci for ci in condition_imgs[condition]], layers=layers, stimuli_identifier="baseline_face." + " ".join(layers)
    )
    condition = 'place'
    scene_activations = alexnet_imagenet()(
        [ci for ci in condition_imgs[condition]], layers=layers, stimuli_identifier="baseline_scene." + " ".join(layers)
    )
    condition = 'body'
    body_activations = alexnet_imagenet()(
        [ci for ci in condition_imgs[condition]], layers=layers, stimuli_identifier="baseline_body." + " ".join(layers)
    )

    condition = 'object'
    object_activations = alexnet_imagenet()(
        [ci for ci in condition_imgs[condition]], layers=layers, stimuli_identifier="baseline_object_." + " ".join(layers)
    )

    cond_activations = {'face': face_activations.values,
                        'place': scene_activations.values,
                        'body': body_activations.values,
                        'object': object_activations.values}


    ## Get predictions for best predicted top-category images

    cond_predicted = {roi: {cond: [] for cond in conditions} for roi in rois}

    for roi in rois:
        for condition in conditions:
            content_predicted = np.matmul(cond_activations[condition], betas[roi])
            print(roi, condition, content_predicted.mean())
            cond_predicted[roi][condition] = content_predicted

    fig, axs =  plt.subplots(ncols=len(rois), figsize=(24, 4))
    for r, roi in enumerate(rois):
        axs[r].title.set_text(roi + " " + "".join(layers))
        sns.distplot(cond_predicted[roi]['face'], hist=False, kde=True,
                     color='r',
                     kde_kws={'linewidth': 3},
                     label="face",
                     ax=axs[r])

        sns.distplot(cond_predicted[roi]['body'], hist=False, kde=True,
                     color='b',
                     kde_kws={'linewidth': 3},
                     label="body",
                     ax=axs[r])
        sns.distplot(cond_predicted[roi]['place'], hist=False, kde=True,
                     color='g',
                     kde_kws={'linewidth': 3},
                     label="scene",
                     ax=axs[r])
        sns.distplot(cond_predicted[roi]['object'], hist=False, kde=True,
                     color='y',
                     kde_kws={'linewidth': 3},
                     label="object",
                     ax=axs[r])
        _ = plt.legend()

    plt.savefig("/home/chan21/projects/visualsemantic/results/encbaseline_" + layers[0] + ".png")


    # Plot mean predicted activations to input images
    plt.figure(figsize=(8, 6))
    plt.title("Mean predicted activations to content input images - " + "".join(layers))
    X = np.array([0, 1, 2, 3])
    plt.bar(X + 0.00,
            [cond_predicted['FFA']['face'].mean(axis=1).mean(),
             cond_predicted['EBA']['face'].mean(axis=1).mean(),
             cond_predicted['PPA']['face'].mean(axis=1).mean(),
             cond_predicted['LOC']['face'].mean(axis=1).mean()],
            color='r',
            width=0.125,
            label='face')

    plt.bar(X + 0.125,
            [cond_predicted['FFA']['body'].mean(axis=1).mean(),
             cond_predicted['EBA']['body'].mean(axis=1).mean(),
             cond_predicted['PPA']['body'].mean(axis=1).mean(),
             cond_predicted['LOC']['body'].mean(axis=1).mean()],
            color='b',
            width=0.125,
            label='body')

    plt.bar(X + 0.25,
            [cond_predicted['FFA']['place'].mean(axis=1).mean(),
             cond_predicted['EBA']['place'].mean(axis=1).mean(),
             cond_predicted['PPA']['place'].mean(axis=1).mean(),
             cond_predicted['LOC']['place'].mean(axis=1).mean()],
            color='g',
            width=0.125,
            label='place')
    plt.bar(X + 0.375,
            [cond_predicted['FFA']['object'].mean(axis=1).mean(),
             cond_predicted['EBA']['object'].mean(axis=1).mean(),
             cond_predicted['PPA']['object'].mean(axis=1).mean(),
             cond_predicted['LOC']['object'].mean(axis=1).mean()
             ],
            color='y',
            width=0.125,
            label='object')

    plt.xticks(X + 0.125, ['FFA', 'EBA', 'PPA', 'LOC'])
    _ = plt.legend()

    plt.savefig("/home/chan21/projects/visualsemantic/results/encbaselineaverage_" + layers[0] + ".png")

    # # Load input images
    # conditions = ["face", "scene", "bodypart", "object"]
    # condition_imgs = {c: [] for c in conditions}
    # for condition in conditions:
    #     stim_path = "../../visualsemantic/stimuli/" + condition
    #     images = listdir(stim_path)
    #     condition_imgs[condition] = [img.split("/")[-1] for img in images]
    #
    # result_path = "../../visualsemantic/results/adversary"
    # results = [item.split("/")[-1] for item in listdir(result_path) if ".png" in item]
    #
    # synthesized_outputs = [item for item in results if len(item.split("_")) > 1]
    # content_inputs = list(
    #     set([item.split("_")[0] + ".png" for item in synthesized_outputs])
    # )
    # style_inputs = list(
    #     set([item.split("_")[1].split(".")[0] + ".png" for item in synthesized_outputs])
    # )
    #
    # # Get activations for input and output images
    # content_activations = alexnet_imagenet(
    #     [os.path.join(result_path, ci) for ci in content_inputs], layers=layers, stimuli_identifier="content." + " ".join(layers)
    # ).values
    #
    # style_activations = alexnet_imagenet(
    #     [os.path.join(result_path, si) for si in style_inputs], layers=layers, stimuli_identifier="style." + " ".join(layers)
    # ).values
    #
    # output_activations = alexnet_imagenet(
    #     [os.path.join(result_path, o) for o in synthesized_outputs], layers=layers, stimuli_identifier="synthesized." + " ".join(layers)
    # ).values
    #
    # # Attack
    #
    # for roi in rois:
    #
    #     content_predicted = {}
    #     style_predicted = {}
    #     output_predicted = {}
    #
    #     nImgsOutput = len(output_activations)
    #     nImgsContent = len(content_activations)
    #     nImgsStyle = len(style_activations)
    #
    #     # normalize and get predicted activations
    #     data = np.vstack([v for v in list(content_activations.values())])
    #     scaler = preprocessing.StandardScaler().fit(data)
    #     data = scaler.transform(data)
    #     for i in range(nImgsContent):
    #         key = list(content_activations.keys())[i]
    #         content_activations[key] = data[i, :]
    #         content_predicted[key] = np.matmul(content_activations[key], betas[roi])
    #
    #     data = np.vstack([v for v in list(style_activations.values())])
    #     scaler = preprocessing.StandardScaler().fit(data)
    #     data = scaler.transform(data)
    #     for i in range(nImgsStyle):
    #         key = list(style_activations.keys())[i]
    #         style_activations[key] = data[i, :]
    #         style_predicted[key] = np.matmul(style_activations[key], betas[roi])
    #
    #     data = np.vstack([v for v in list(output_activations.values())])
    #     scaler = preprocessing.StandardScaler().fit(data)
    #     data = scaler.transform(data)
    #     for i in range(nImgsOutput):
    #         key = list(output_activations.keys())[i]
    #         output_activations[key] = data[i, :]
    #         output_predicted[key] = np.matmul(output_activations[key], betas[roi])
    #
    #     # create rdms
    #     rdm_output = np.zeros((nImgsOutput, nImgsOutput))
    #     rdm_content = np.zeros((nImgsOutput, nImgsOutput))
    #     rdm_style = np.zeros((nImgsOutput, nImgsOutput))
    #
    #     for i, c in enumerate(sorted(output_activations.keys())):
    #         for j, d in enumerate(sorted(output_activations.keys())):
    #
    #             r, p = stats.pearsonr(output_predicted[c], output_predicted[d])
    #             # r = 1 - spatial.distance.cosine(style_activations[c], style_activations[d])
    #             rdm_output[i, j] = r
    #
    #     for i, c in enumerate(sorted(output_activations.keys())):
    #         for j, d in enumerate(sorted(output_activations.keys())):
    #             file_c = c.split("_")[0] + ".png"
    #             file_d = d.split("_")[0] + ".png"
    #             r, p = stats.pearsonr(
    #                 content_predicted[file_c], content_predicted[file_d]
    #             )
    #             # r = 1 - spatial.distance.cosine(content_activations[c], content_activations[d])
    #             rdm_content[i, j] = r
    #
    #     for i, c in enumerate(sorted(output_activations.keys())):
    #         for j, d in enumerate(sorted(output_activations.keys())):
    #             file_c = c.split("_")[1].split(".")[0] + ".png"
    #             file_d = d.split("_")[1].split(".")[0] + ".png"
    #             r, p = stats.pearsonr(style_predicted[file_c], style_predicted[file_d])
    #             # r = 1 - spatial.distance.cosine(style_activations[c], style_activations[d])
    #             rdm_style[i, j] = r
    #
    #     triu_ind = np.triu_indices(nImgsOutput, k=1)
    #     r, p = stats.spearmanr(
    #         rdm_output[triu_ind], rdm_content[triu_ind], axis=None
    #     )  # compute rank correlation between each model and each roi rdm
    #     rsa[roi]["content"] = r
    #     r, p = stats.spearmanr(rdm_output[triu_ind], rdm_style[triu_ind], axis=None)
    #     rsa[roi]["style"] = r
    #     print(rsa)
    #
    # f = open("../../visualsemantic/results/rsa/" + " ".join(layers) + ".pkl", "wb")
    # pickle.dump(rsa, f)
    # f.close()
