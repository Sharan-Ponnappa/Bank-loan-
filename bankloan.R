rm(list = ls())

setwd("F:/proj2")
#library list
libs=c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced","C50","gridExtra","class")

#load Libraries
lapply(libs,require,character.only=TRUE)


#read data
raw_dat=read.csv("bank-loan.csv",header = T,na.strings = c(" ","","NA"))       

#convert ED and Default  into factor as it is catogorical variable
raw_dat$ed=factor(raw_dat$ed)

raw_dat$default=factor(raw_dat$default)


#outlier analysis

indx_val = sapply(raw_dat, is.numeric)
num_data = raw_dat[,indx_val]

cols= colnames(num_data)

cols

#plot ggplot
for (i in 1:length(cols)) {
  assign(paste0("plot",i),ggplot(aes_string(y=(cols[i]),x="default"),dat=subset(raw_dat))+
           stat_boxplot(geom='errorbar',width=0.5)+
           geom_boxplot(outlier.colour="red",fill="blue",outlier.shape=18,outlier.size=1,notch=F)+
           theme(legend.position="bottom")+
           labs(y=cols[i],x="default"))
}

gridExtra::grid.arrange(plot1,plot2,plot3,ncol=3)


#remove outliers
for (i in cols) {
  
  val=raw_dat[,i][raw_dat[,i]%in%boxplot.stats(raw_dat[,i])$out]
  raw_dat[,i][raw_dat[,i]%in%val]=NA
}

"

#mean income=	39.14088 creddebt=	1.080226  ;debtinc=9.619589 ;otherdebt=	2.350541

for (i in cols) {
  raw_dat[,i][is.na(raw_dat[,i])]=mean(raw_dat[,i],na.rm = T)
}
 

#median income=33 creddebt=		0.798002  ;debtinc=	8.5 ;otherdebt=1.808640
for (i in cols) {
  raw_dat[,i][is.na(raw_dat[,i])]=median(raw_dat[,i],na.rm = T)
}
"
" We select KNN as it gives the nearest values:
#income-    5=28       :  16.35523        15=	100      :  75.59320
#creddebt-  5=1.787436 :  0.4802621       18=1.181952  :  1.5572370
#otherdebt- 9=3.277652 :  2.134183        15=5.396300  :  0.509184
#debtinc-   9=	24.4   :  18.707526       18=	7.6      :  6.634931
" 
raw_dat=knnImputation(raw_dat,k=3)

##

corrgram(raw_dat[,indx_val],order = F,upper.panel = panel.shade, text.panel = panel.text,main="correlation")


fact_indx=sapply(raw_dat,is.factor)
fact_data=raw_dat[,fact_indx]

#chi square test
print(chisq.test(table(raw_dat$default,raw_dat$ed)))

#normality check
qqnorm(raw_dat$creddebt)
hist(raw_dat$creddebt)


#normalization
for (i in cols) {
  raw_dat[,i]=(raw_dat[,i]-min(raw_dat[,i]))/(max(raw_dat[,i])-min(raw_dat[,i]))
}

#standardization

for (i in cols) {
  raw_dat[,i]=(raw_dat[,i]-mean(raw_dat[,i]))/sd(raw_dat[,i])
}

#machine learning
train_index=createDataPartition(raw_dat$default,p=.80,list=F)
train=raw_dat[train_index,]
test=raw_dat[-train_index,]

#logistic Reggression model
LGmodl=glm(default ~.,train,family = binomial)
summary(LGmodl)
LG_Pred=predict(LGmodl,newdata = test,type="response")

LG_Pred=ifelse(LG_Pred>0.5,1,0)

conf_mattLG=table(test$default,LG_Pred)
confusionMatrix(conf_mattLG)
"acc=81
fp=9 =6%
fn=18=55%"

# KNN
library(class)

KNN_pred=knn(train[,1:8],test[,1:8],train$default,k=5)

conf_mattKNN=table(test$default,LG_Pred)
confusionMatrix(conf_mattKNN)


"acc=81
fp=9 =6%
fn=18=55%"

