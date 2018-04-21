library(data.table)
library(lightgbm)
setwd("~/mycloud/repos/useR_meetup_2018_04")

###########
## Simulate simple linear data
set.seed(0)
n = 500
x = seq(0, 100, length.out = n)

data_linear = data.frame(
    x = x,
    y = 0.5*x + rnorm(n, 0, 10)
)
X = as.matrix(data_linear[, 'x', drop = FALSE])

# A couple of functions, which assume the existence of 
#   X, y in the parent environment!
mygbm = function(num_leaves = 5, learning_rate = 1, nrounds = 1,
                 min_data_in_leaf = 1){
    return(lightgbm(
        data = X,
        label = data_linear$y,
        num_leaves = num_leaves,
        min_data_in_leaf = min_data_in_leaf,
        learning_rate = learning_rate,
        nrounds = nrounds,
        objective = "regression"))
}
plot_pdf = function(filename, yhat = NULL, main = ''){
    file = paste0("images/", filename, ".pdf")
    pdf(file, width = 6, height = 5)
    plot(data_linear, pch = 16, cex = 0.4, bty = 'n', main = main)
    if(!is.null(yhat)){
        lines(x = data_linear$x, y = yhat, lwd = 2)
    }
    dev.off()
}

# Display the data
plot_pdf("data_linear",
         main = 'Simulated y = 0.5*x + rnorm(n, 0, 10)')


###########
## Demonstrate LightGBM's fit to the data
num_leaves = 4
bst = mygbm(num_leaves = num_leaves)
plot_pdf(paste0("lgb_fit_", num_leaves, "_leaves"), 
         yhat = predict(bst, data = X))


###########
## Stumble around a little, trying various hyperparameters
num_leaves = 20
bst = mygbm(num_leaves = num_leaves)
plot_pdf(paste0("lgb_fit_", num_leaves, "_leaves"), 
         yhat = predict(bst, data = X))

learning_rate = 0.3
bst = mygbm(num_leaves = num_leaves, learning_rate = learning_rate)
plot_pdf(paste0("lgb_fit_rate_", learning_rate), 
         yhat = predict(bst, data = X))

nrounds = 100
bst = mygbm(num_leaves = num_leaves, learning_rate = learning_rate,
            nrounds = nrounds)
plot_pdf(paste0("lgb_fit_nrounds_", nrounds), 
         yhat = predict(bst, data = X))


###########
## Getting tired of this ... time for automatic hyperparameter tuning
library(mlrMBO)
configureMlr(on.learner.warning = "quiet", show.learner.output = FALSE)

y = data_linear$y

search_space = makeParamSet(
    makeIntegerParam("num_leaves", 2, 30),
    makeIntegerParam("min_data_in_leaf", 1, 50),
    makeNumericParam("learning_rate", 0.001, 0.4),
    makeIntegerParam("nrounds", 20, 200)
)

objective = function(h){
    cv = lightgbm::lgb.cv(
        data = X, label = y,
        num_leaves = h[1],
        min_data_in_leaf = h[2],
        learning_rate = h[3],
        nrounds = h[4],
        nfold = 5, verbose = -1, objective = "regression")
    out = tail(cv$record_evals$valid$l2$eval, 1)[[1]]
    if((!is.numeric(out))||(length(out) != 1)) browser()
    return(out)
}

tuning_objective = makeSingleObjectiveFunction(
    name = "wrap_lightgbm",
    fn = objective,
    par.set = search_space,
    noisy = TRUE,
    minimize = TRUE
)

ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, time.budget = 3600)
tuning_cache = "tune_res.RDS"
if(!file.exists(tuning_cache)){
    res = mbo(tuning_objective, control = ctrl, show.info = FALSE)
    saveRDS(res, "tune_res.RDS")
}
res = readRDS("tune_res.RDS")

bst = lightgbm(
    data = X,
    label = data_linear$y,
    num_leaves = res$x$num_leaves,
    min_data_in_leaf = res$x$min_data_in_leaf,
    learning_rate = res$x$learning_rate,
    nrounds = res$x$nrounds,
    objective = "regression")

plot_pdf(paste0("lgb_tuned"), 
         yhat = predict(bst, data = X))

unlink('lightgbm.model')
