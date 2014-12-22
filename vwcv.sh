cat 23_24.vwapp  25_26.vwapp  27_28.vwapp  29_30.vwapp|vw --passes 3 -c --loss_function logistic --link=logistic -f 21_22.model
vw -i 21_22.model -t 21_22.vwapp -p 21_22.out --loss_function logistic
cat 21_22.vwapp  25_26.vwapp  27_28.vwapp  29_30.vwapp|vw --passes 3 -c --loss_function logistic --link=logistic -f 23_24.model
vw -i 23_24.model -t 23_24.vwapp  -p 23_24.out --loss_function logistic
cat 21_22.vwapp  23_24.vwapp  27_28.vwapp  29_30.vwapp|vw --passes 3 -c --loss_function logistic --link=logistic -f 25_26.model
vw -i 25_26.model -t 25_26.vwapp  -p 25_26.out --loss_function logistic
cat 21_22.vwapp  23_24.vwapp  25_26.vwapp  29_30.vwapp|vw --passes 3 -c --loss_function logistic --link=logistic -f 27_28.model
vw -i 27_28.model -t 27_28.vwapp  -p 27_28.out --loss_function logistic
cat 21_22.vwapp  23_24.vwapp  25_26.vwapp  27_28.vwapp|vw --passes 3 -c --loss_function logistic --link=logistic -f 29_30.model
vw -i 29_30.model -t 29_30.vwapp  -p 29_30.out --loss_function logistic

