res = stats.ttest_rel(knn_scores , naive_bayes_scores , alternative = "greater")
print("Not scaled: knn > naive_bayes? pval=", res.pvalue)

res = stats.ttest_rel(knn_scores_minmax , gnb_scores_minmax , alternative = "greater")
print("Scaled: knn > naive_bayes? pval=", res.pvalue)