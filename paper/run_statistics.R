# Calculates the statistics reported in the paper

library(lme4)
library(lmerTest)
library(car)

print("")
print("Compare cross-validation results between models")
print("")

csvpath = 'configs/triplets3ab_crossval/results_by_fold_log2.csv'
df = read.csv(csvpath,stringsAsFactors=TRUE)

fitF <- lmer(test_loss ~ variables_item*variables_rater + (1|fold), data=df)
anova(fitF)

for (design_speaker in levels(df$variables_item)) {
    print("")
    print(design_speaker)
    data = df[df$variables_item==design_speaker,]
    fitF = lmer(test_loss ~ variables_rater + (1|fold), data=data)
    print(anova(fitF))
}

l_n = df[c(1,2,3,4,5),]$test_loss
l_gs = df[c(16,17,18,19,20),]$test_loss
l_g = df[c(11,12,13,14,15),]$test_loss
l_s = df[c(6,7,8,9,10),]$test_loss
l_gsi = df[c(26,27,28,29,30),]$test_loss
l_gsir = df[c(56,57,58,59,60),]$test_loss

t.test(l_n, l_gs, paired=TRUE)
t.test(l_g, l_gs, paired=TRUE)
t.test(l_s, l_gs, paired=TRUE)
t.test(l_gs, l_gsi, paired=TRUE)
t.test(l_gsi, l_gsir, paired=TRUE)

print("")
print("Regression models for speaker scores separated by feature")
print("")

csvpath = 'configs/triplets3ab_crossval/best/speaker_scores.csv'
df = read.csv(csvpath,stringsAsFactors=TRUE)

print("Feature 1-2: Nativeness")
print("")
fitF = lm(score~nativeness*rating_nativeness, data=df[df$feature=="emb0",])
Anova(fitF)
fitF = lm(score~nativeness*rating_nativeness, data=df[df$feature=="emb1",])
Anova(fitF)

print("")
print("Feature 3: Gender")
print("")
fitF = lm(score~sex*rating_gender, data=df[df$feature=="emb2",])
Anova(fitF)

print("")
print("Feature 4-5: Accent")
print("")
fitF = lm(score~usuk*rating_usuk, data=df[df$feature=="emb3",])
Anova(fitF)
fitF = lm(score~usuk*rating_usuk, data=df[df$feature=="emb4",])
Anova(fitF)

print("")
print("Anovas for rater weights")
print("")

csvpath = 'configs/triplets3ab_crossval/best/rater_weights.csv'
df = read.csv(csvpath,stringsAsFactors=TRUE)
df = df[seq(1,300),] # First five features

fitF <- lmer(weight ~ group*sex*feature + (1|participant_id), data=df)
anova(fitF)

for (feature in c('emb0','emb1','emb2','emb3','emb4')) {
    print("")
    print(feature)
    data = df[(df$feature==feature),]
    fitF = lm(weight ~ group*sex, data=data)
    print(Anova(fitF))
}

data = df[(df$feature=='emb1')&(df$sex=="female"),]
fitF = lm(weight ~ group, data=data)
print(Anova(fitF))

data = df[(df$feature=='emb1')&(df$sex=="male"),]
fitF = lm(weight ~ group, data=data)
print(Anova(fitF))

print("")
print('Model comparison - nativeness ratings')
print("")

csvpath = 'configs/triplets3ab_crossval_rating/uniques_individual.csv'
df = read.csv(csvpath,stringsAsFactors=TRUE)
df$rater = as.factor(df$rater)
df$nativeness = 'nonnative'
df[df$group=='english_uk',]$nativeness = 'native'

fitF <- lmer(bits ~ group*partition + (1|rater), data=df)
anova(fitF)

fitF <- lm(bits ~ group, data=df[df$partition=='group_unique',])
Anova(fitF)

fitF <- lm(bits ~ group, data=df[df$partition=='rating_unique',])
Anova(fitF)
