import os
from scipy import stats
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as tr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from tqdm import tqdm

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [
        f
        for f in files
        if (f != ".DS_Store" and f != "._.DS_Store" and f != ".ipynb_checkpoints")
    ]
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def p2r(p, n):
    t = stats.t.ppf(1 - p, n - 2)
    r = (t**2 / ((t**2) + (n - 2))) ** 0.5
    return r


def image_to_tensor(image, resolution=None, do_imagenet_norm=True):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if resolution is not None:
        image = tr.resize(image, resolution)
    if image.width != image.height:  # if not square image, crop the long side's edges
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    image = tr.to_tensor(image)
    if do_imagenet_norm:
        image = imagenet_norm(image)
    return image


def imagenet_norm(image):
    dims = len(image.shape)
    if dims < 4:
        image = [image]
    image = [tr.normalize(img, mean=imagenet_mean, std=imagenet_std) for img in image]
    image = torch.stack(image, dim=0)
    if dims < 4:
        image = image.squeeze(0)
    return image


def imagenet_unnorm(image):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    image = image.cpu()
    image = image * std + mean
    return image


def cv_regression(condition_features, subject, l2=0.0, random=None):
    # Get cross-validated mean test set correlation
    rs = []

    for test_conditions in subject.cv_sets:
        train_conditions = [c for c in subject.conditions if c not in test_conditions]
        train_features = np.stack([condition_features[c] for c in train_conditions])
        test_features = np.stack([condition_features[c] for c in test_conditions])
        train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
        test_voxels = np.stack([subject.condition_voxels[c] for c in test_conditions])

        train_mask = ~np.isnan(train_voxels)
        test_mask = ~np.isnan(test_voxels)

        train_voxels = np.stack(
            [
                train_voxels[i, :][train_mask[i, :]]
                for i in range(0, train_voxels.shape[0])
            ]
        )
        test_voxels = np.stack(
            [test_voxels[i, :][test_mask[i, :]] for i in range(0, test_voxels.shape[0])]
        )

        _, r = regression(
            train_features, train_voxels, test_features, test_voxels, l2=l2
        )
        rs.append(r)
    mean_r = np.mean(rs)

    # Train on all of the data
    train_conditions = subject.conditions
    train_features = np.stack([condition_features[c] for c in train_conditions])
    train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])

    train_mask = ~np.isnan(train_voxels)
    train_voxels = np.stack(
        [train_voxels[i, :][train_mask[i, :]] for i in range(0, train_voxels.shape[0])]
    )

    weights = regression(
        train_features, train_voxels, None, None, l2=l2, validate=False
    )
    return weights, mean_r


def cv_regression_w(
    features,
    wordembedding,
    fit=None,
    k=9,
    l2=0.0,
    pc_fmri=None,
    pc_embedding=None,
    shuffle=None,
):
    if pc_fmri is not None:
        pca_fmri = PCA(n_components=pc_fmri)
    if pc_embedding is not None:
        pca_embedding = PCA(n_components=int(pc_embedding))

    fold_size = len(wordembedding) / k

    kf = KFold(n_splits=k)
    rs = []
    for train_index, test_index in kf.split(features):
        if shuffle is not None:
            shuffle_array = shuffle.split(",")
            shuffle_array = [int(st) for st in shuffle_array]

            shuffle_train = [
                i - len(test_index) for i in shuffle_array if i >= len(test_index)
            ]
            shuffle_test = [i for i in shuffle_array if i < len(test_index)]
        else:
            shuffle_train = np.arange(len(train_index))
            shuffle_test = np.arange(len(test_index))

        train_features = features[
            train_index,
        ]
        test_features = features[
            test_index,
        ]
        if pc_fmri is not None:
            pca_fmri.fit(train_features)
            train_features = pca_fmri.transform(train_features)
            test_features = pca_fmri.transform(test_features)

        train_embeddings = np.stack(
            [
                embedding
                for i, embedding in enumerate(wordembedding.values())
                if i in train_index
            ]
        )
        train_embeddings = train_embeddings[shuffle_train]

        test_embeddings = np.stack(
            [
                embedding
                for i, embedding in enumerate(wordembedding.values())
                if i in test_index
            ]
        )
        test_embeddings = test_embeddings[shuffle_test]

        if pc_embedding is not None:
            pca_embedding.fit(train_embeddings)
            train_embeddings = pca_embedding.transform(train_embeddings)
            test_embeddings = pca_embedding.transform(test_embeddings)

        weights, r = regression(
            train_features, train_embeddings, test_features, test_embeddings, l2=l2
        )
        rs.append(r)
    rs = np.array(rs)
    mean_r = np.nanmean(rs, axis=0)  # mean across k folds
    return weights, mean_r


def regression(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    r_ = []
    if validate:
        y_pred = regr.predict(x_test)
        y_pred = y_pred.transpose()
        y_test = y_test.transpose()
        for (y_t, y_p) in zip(y_test, y_pred):
            r = correlation(y_t, y_p)
            r_.append(r)
        return weights, r_
    else:
        return weights


def correlation(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean()
    return r
