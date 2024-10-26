job = [col for col in X_normalized.columns if col.startswith("job_")]
education = [col for col in X_normalized.columns if col.startswith("education_")]

melted_jobs = pd.melt(X, id_vars = "cluster", value_vars = job, var_name = "job", value_name = "value")
melted_jobs["job"] = melted_jobs["job"].str.replace("job_", "")

melted_education  =  pd.melt(X, id_vars = "cluster", value_vars = education, var_name = "education", value_name = "value")
melted_education["education"] = melted_education["education"].str.replace("education_", "")

sns.displot(
    data = melted_jobs[melted_jobs["value"]], 
    y = "job",
    hue = "cluster",
    multiple = "dodge",
    stat = "density",
    shrink = 0.8,
    common_norm = False,
)

sns.displot(
    data = melted_education[melted_education["value"]],  
    y = "education",
    hue = "cluster",
    multiple="dodge",
    stat = "density",
    shrink = 0.8,
    common_norm = False,
)