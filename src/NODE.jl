using NPZ
using OrdinaryDiffEq
using Flux

# We open the reduced state previously prepared in Python
reducedState = npzread("data/processed/sstReducedStateCOPERNICUS20102019Prepared.npz")

# for key in keys(reducedState) 
#     println(key)
# end

# println(reducedState["tVal"])

# We set a shape of struct adapted to the input reduced state
struct inputVars
    PCsTrain
    PCsVal
    tTrain
    tVal
end

# We instance the retrieve .npz reduced state
iV = inputVars(reducedState["PCsTrain"], reducedState["PCsVal"], reducedState["tTrain"], reducedState["tVal"])

println(size(iV.PCsTrain)) # (2922,150)
println(size(iV.PCsVal)) # (730, 150)

# We remerge initial unsplitted PCs state
PCs = cat(iV.PCsTrain, iV.PCsVal, dims=1)
dT = iV.tTrain[2]-iV.tTrain[1] # We assume time step as already been normalized
spanT = size(PCs)[1]

println(spanT) # 3652
println(dT) # 1.0
println(size(PCs)) # (3652, 150) -> correct because we compute 150 mods and we have 10 years of data with one point for each day

# From now on, we desire to implement Neural ODE network

# Before we are splitting dataset between a train set and a validation set
ratioTrain = 0.7 # We set the ratio of the set we want to train (so we can easily modify it later)
zTrain, zTest = PCs[1:Int32.(round(ratioTrain*spanT)), :], PCs[Int32.(round(ratioTrain*spanT)):spanT, :] # Here we separete train and validation dataset

# println(size(zTrain))
# println(size(zTest))

# Now we can start to set the NN and its parameters
z0 = zTrain[1]






