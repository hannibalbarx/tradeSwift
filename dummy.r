test.file="p10.t.site_train.1"
train.file="p10.t.site_train.2"
out.file="p10.t.site_train.2.dummy100"
threshold=100
test.cols=c('id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21')

cols=c('id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21')
test=read.csv(test.file,sep=",",col.names=test.cols,colClasses="factor")
test$click<-as.integer(test$click)-1
test.di=aggregate(test$click,list(test$device_ip),length)
test.di<-test.di[order(test.di$x,decreasing = TRUE),]

train=read.csv(train.file,sep=",",col.names=cols,colClasses="factor")
train$click<-as.integer(train$click)-1
train.di=aggregate(train$click,list(train$device_ip),length)
train.di<-train.di[order(train.di$x,decreasing = TRUE),]

n.di=merge(train.di,test.di,by="Group.1",all.x=TRUE)
n.di$ndi<-as.character(n.di$Group.1)
n.di[is.na(n.di$x.y),4]<-"dummy"
n.di$ndi=as.factor(n.di$ndi)
n.di[n.di$x.x<threshold,4]<-"dummy"
train2=merge(train,n.di[,c("Group.1","ndi")],by.x="device_ip", by.y="Group.1")
train2$device_ip<-train2$ndi
write.csv(train2[,cols],file=out.file,quote=FALSE,row.names=FALSE)

#dummy

setwd("/home/ubuntu/data/by_day")
test.file="141029.app_train"
train.file="21_28.app_train"
cols=c('id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21')
colsC=rep('factor',length(cols))
colsC[2]<-"integer"

test=read.csv(test.file,sep=",",header=F, col.names=cols,colClasses=colsC)
test$rowid<-1:nrow(test)

train=read.csv(train.file,sep=",",header=F, col.names=cols,colClasses=colsC)
train$rowid<-1:nrow(train)

#fields=data.frame(test$id)
#fields$click<-test$click
#fields$hour<-as.integer(substring(test$hour,7,8))
fields=data.frame(
  paste(test$click,as.integer(substring(test$hour,7,8),' ')))
  
#fields_t=data.frame(train$id)
#fields_t$click<-train$click
#fields_t$hour<-as.integer(substring(train$hour,7,8))
fields_t=data.frame(
  paste(train$click,as.integer(substring(train$hour,7,8),' ')))

feature_n=1
for (i in c(1,2,6:21)){print (i)
  fn=3+ i
  #fn=3+ 1
  
  l=data.frame(levels(test[[fn]]))

  l$rowid<-feature_n:(feature_n+nrow(l)-1)
  feature_n=feature_n+nrow(l)
  
  f<-merge(data.frame(test[[fn]], test['rowid']),l,by.x=1,by.y=1)
  f<-f[order(f$rowid.x),'rowid.y']
  fields=data.frame(fields, f)
  names(fields)[ncol(fields)]<-paste('field',fn-3,sep='')
  
  f_t<-merge(data.frame(train[[fn]], train['rowid']),l,by.x=1,by.y=1,all.x=T)
  if (nrow(f_t[is.na(f_t$rowid.y),])>0) {
    f_t[is.na(f_t$rowid.y),]$rowid.y<-feature_n
    feature_n=feature_n+1
  }
  
  f_t<-f_t[order(f_t$rowid.x),'rowid.y']
  fields_t=data.frame(fields_t, f_t)
  names(fields_t)[ncol(fields_t)]<-paste('field',fn-3,sep='')
}

write.table(fields,file="141029.du.app_train",
            quote=F, sep=",",col.names=F,row.names=F)
write.table(fields_t,file="21_28.du.app_train",
            quote=F, sep=",",col.names=F,row.names=F)


sed -i 's/,/:1 /g' 21_28.du.app_train
sed -i 's/$/:1/g' c
sed -i 's/e+05/00000/g' 21_28.du.app_train

