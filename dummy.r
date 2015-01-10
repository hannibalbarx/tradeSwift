test.file="p10.t.site_train.2"
train.file="p10.t.site_train.1"
out.file="p10.t.site_train.1.dummy50"
threshold=50

cols=c('id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21')
test=read.csv(test.file,sep=",",col.names=cols,colClasses="factor")
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
