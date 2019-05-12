from shutil import copyfile
import os

os.chdir('../')
projDir = os.getcwd()
assert(os.path.isdir(projDir))

print(os.getcwd())
inputBase = "modelOutput" #str(raw_input("base input dir  "))
model = "InceptionV3" #str(raw_input("model  "))
outputBase = "Evaluation/modelVariance" #str(raw_input("output Dir  "))

print(inputBase)
print(model)
print(outputBase)

subsampleExperimentDir = os.path.join(projDir, inputBase, "subsample_" + model)

if not os.path.isdir(subsampleExperimentDir):
    raise Exception('error: '+subsampleExperimentDir)
assert(os.path.isdir(outputBase))
sampleDirs = os.listdir(subsampleExperimentDir)

for i, smpl in enumerate(sampleDirs):
    samplePath = os.path.join(subsampleExperimentDir, smpl)
    assert(os.path.isdir(samplePath))
    modelDir = [f for f in os.listdir(samplePath) if model in f]
    assert(len(modelDir)==1)
    modelHistPath = os.path.join(samplePath, modelDir[0], "modelHistory.pickle")
    outputPath = os.path.join(projDir, outputBase, model, "modelHistory_{}.pickle".format(i))
    copyfile(modelHistPath, outputPath)