import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

train = pd.read_csv('data/train_label.csv')
fea = pd.read_csv('data/train_base.csv')
train = pd.merge(train, fea, on='user', how='left')

# This way we have randomness and are able to reproduce the behaviour within this cell.
np.random.seed(13)


def mean_coding(data, feature, target='label'):
    n_folds = 10
    n_inner_folds = 5
    mean_coded = pd.Series()

    # 所有数据的label均值
    default_mean = data[target].mean()
    kf = KFold(n_splits=n_folds, shuffle=True)

    out_mean_cv = pd.DataFrame()
    split = 0
    # 对所有数据做CV
    for in_fold, out_fold in kf.split(data[feature]):

        impact_coded_cv = pd.Series()
        kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
        inner_split = 0
        inner_mean_cv = pd.DataFrame()

        # in_fold数据的label均值
        default_inner_mean = data.iloc[in_fold][target].mean()

        # 对in_fold数据做CV
        for in_fold_inner, out_fold_inner in kf_inner.split(data.iloc[in_fold]):
            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)

            # 对in_fold_inner做group_by求出mean:feature-mean
            in_fold_inner_mean = data.iloc[in_fold_inner].groupby(by=feature)[target].mean()

            # in_fold的mean使用cv后的in_fold_inner的mean值，如果没有在里面，就用in_fold的label均值
            impact_coded_cv = impact_coded_cv.append(data.iloc[in_fold].apply(
                lambda x: in_fold_inner_mean[x[feature]]
                if x[feature] in in_fold_inner_mean.index
                else default_inner_mean
                , axis=1))

            # Also populate mapping (this has all group -> mean for all inner CV folds)
            inner_mean_cv = inner_mean_cv.join(pd.DataFrame(in_fold_inner_mean), rsuffix=inner_split, how='outer')
            inner_mean_cv.fillna(value=default_inner_mean, inplace=True)
            inner_split += 1

        # Also populate mapping
        out_mean_cv = out_mean_cv.join(pd.DataFrame(inner_mean_cv), rsuffix=split, how='outer')
        out_mean_cv.fillna(value=default_mean, inplace=True)
        split += 1

        mean_coded = mean_coded.append(data.iloc[out_fold].apply(
            lambda x: inner_mean_cv.loc[x[feature]].mean()
            if x[feature] in inner_mean_cv.index
            else default_mean
            , axis=1))

    return mean_coded, out_mean_cv.mean(axis=1), default_mean


# Apply the encoding to training and test data, and preserve the mapping
impact_coding_map = {}
for f in ['city']:
    print("Impact coding for {}".format(f))
    train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = mean_coding(train, f)
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_means = impact_coding_map[f]
    print(123)
    # test_data["impact_encoded_{}".format(f)] = test_data.apply(
    #     lambda x: mapping[x[f]] if x[f] in mapping else default_mean, axis=1)
