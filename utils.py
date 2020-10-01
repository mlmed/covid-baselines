import numpy as np
import pandas as pd
import sklearn, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.neural_network
import scipy, scipy.stats, collections


def evaluate(data, labels, title, regions, test_region, plot=True, seed=0, method="linear", target_str="", usemlp=False):
    X = data
    y = labels.astype(float)
    
#     gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.5,test_size=0.5, random_state=seed)
#     train_inds, test_inds = next(gss.split(X, y, groups))
    test_region_mask = (regions == test_region)
    train_inds = np.where(~test_region_mask)[0]
    test_inds = np.where(test_region_mask)[0]

    X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], y.iloc[train_inds], y.iloc[test_inds]
    #print("X_train", X_train.shape, "X_test", X_test.shape)

    res = {}
    
    if method=="linear":
        model = sklearn.linear_model.LinearRegression()
    if (method=="logistic") and not usemlp:
        model = sklearn.linear_model.LogisticRegression()
    if usemlp:    
        model = sklearn.neural_network.MLPClassifier(random_state=seed, early_stopping=True)
        
        
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    pdiff = y_test - y_pred
    
    
    res["name"] = title
    res["test_region"] = test_region
    if method=="linear":
        res["R^2"] = sklearn.metrics.r2_score(y_test, y_pred)
        res["Correlation"] = scipy.stats.pearsonr(y_test,y_pred)[0]
        if np.isnan(res["Correlation"]): res["Correlation"] = 0
        res["MAE"] = sklearn.metrics.mean_absolute_error(y_test, y_pred)
        #res["MSE"] = sklearn.metrics.mean_squared_error(y_test, y_pred)
        res["# test samples"] = int(len(y_test))
        
    if method=="logistic":
        res["AUROC"] = sklearn.metrics.roc_auc_score(y_test, y_pred)
        res["AUPRC"] = sklearn.metrics.average_precision_score(y_test, y_pred)
        res["# test samples"] = collections.Counter(y_test == 1)
    
    if usemlp:
        #res["# params"] = "{}+{}".format(len(np.concatenate([a.flatten() for a in model.coefs_])), len(np.concatenate([a.flatten() for a in model.intercepts_])))
        res["# params"] = "{}".format(len(np.concatenate([a.flatten() for a in model.coefs_]))+len(np.concatenate([a.flatten() for a in model.intercepts_])))
        res["method"] = "MLP"
        res["name"] = res["name"] + " (MLP)"
    else:
        res["# params"] = "{}+1".format(len(model.coef_.flatten()))
        res["method"] = method

    
    if plot:
        fig, ax = plt.subplots(figsize=(6,4), dpi=100)
        for x,y,yp in zip(y_test,y_test,y_pred):
            plt.plot((x,x),(y,yp),color='red',marker='')

        pmax = int(np.max([y_pred.max(), y_test.max()]))+2
        plt.plot(range(pmax),range(pmax), c="gray", linestyle="--")
        plt.xlim(0,pmax-1)
        plt.ylim(0,pmax-1)

        plt.scatter(y_test, y_pred);
        plt.ylabel("Model prediction ($y_{pred}$)")
        plt.xlabel("Ground Truth ($y_{true}$)")
        plt.title(title);
        plt.text(0.01,0.97, "$R^2$={0:0.2f}".format(res["R^2"])+ "\n"+ 
                 "Correlation={0:0.2f}".format(res["Correlation"]), ha='left', va='top', transform=plt.gca().transAxes)

    return res#, pdiff
