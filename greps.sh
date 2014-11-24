block 1
egrep ',[01],1410(21(00|01|12|16|17|18|19|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(22(00|01|02|03|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(23(00|01|02|06|12|13|14|19|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(24(00|01|02|03|04|05|09|10|11|19|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(25(00|01|02|03|04|05|06|07|19|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(26(00|01|02|03|18|19|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(27(00|01|02|03|04|08|09|10|11|12|13|14|15|19|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(28(00|01|02|03|04|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(29(00|01|02|04|05|06|17|18|19|20|21|22|23)),' site_train>>tmp
egrep ',[01],1410(30(00|01|02|03|04|05|06|07|17|18|19|20|21|22|23)),' site_train>>tmp

block 2
egrep ',[01],1410(21(02|03|07|08|09|10|11|13|14|15)),' site_train>>tmp
egrep ',[01],1410(22(04|14|15|16|17|18|19)),' site_train>>tmp
egrep ',[01],1410(23(03|07|10|11|15|16|17|18)),' site_train>>tmp
egrep ',[01],1410(24(06|07|08|12|13|14|15|18)),' site_train>>tmp
egrep ',[01],1410(25(08|09|10|11|12|18)),' site_train>>tmp
egrep ',[01],1410(26(04|05|06|08|17)),' site_train>>tmp
egrep ',[01],1410(27(05|06|07|16|18)),' site_train>>tmp
egrep ',[01],1410(28(05|12|19)),' site_train>>tmp
egrep ',[01],1410(29(03|07|08|09|11|12|13|15|16)),' site_train>>tmp
egrep ',[01],1410(30(08|09|10|11|13|15|16)),' site_train>>tmp

block 3
egrep ',[01],1410(21(04|05|06)),' site_train>>tmp
egrep ',[01],1410(22(05|06|07|08|09|10|11|12|13)),' site_train>>tmp
egrep ',[01],1410(23(04|05|08|09)),' site_train>>tmp
egrep ',[01],1410(24(16|17)),' site_train>>tmp
egrep ',[01],1410(25(13|14|15|16|17)),' site_train>>tmp
egrep ',[01],1410(26(07|09|10|11|12|13|14|15|16)),' site_train>>tmp
egrep ',[01],1410(27(17)),' site_train>>tmp
egrep ',[01],1410(28(06|07|08|09|10|11|13|14|15|16|17|18)),' site_train>>tmp
egrep ',[01],1410(29(10|14)),' site_train>>tmp
egrep ',[01],1410(30(12|14)),' site_train>>tmp


site_test
egrep ',141031(23|00|22|01|21|03),' site_test>>site_test_1.3
egrep ',141031(02|20|04|19|05|12),' site_test>>site_test_2.3
egrep ',141031(08|17|10|18|09|14|11|13|06|16|07|15),' site_test>>site_test_3.3


head -1 train>250.train
cp 250.train 250.valid
egrep ',[01],1410(2808|2816|2810|2811|2416|2304|2817|2812),' train>>250.train
egrep ',[01],1410(2815|2104|2417|2206|2205|3013|2105|3014),' train>>250.valid

egrep ',[01],1410(2900|3002|2923|2921|2722|2401|2721|2223|2521|3001|2922|2323|2522|2121|2300|2622|2702|3023|2800|2400|2621|2601|2501|2200|2504|2623|2523|2701|2123|2500|2600|2503|2700|2506|2502|2419|2423|2420|2421|2422),' train>>100.train
egrep ',[01],1410(2122|3000|2723),' train>>100.valid

egrep ',[01],1410(2919|2402|2822|2403|2221|2703|2803|2901|2711|2505|2920|2321|3021|2603|2120|3020|2222|2720|2322|2714|2715|2202|2801|2301|2620|3022|2710|2520|2602|2201|3002|2923|2921|2722|2401|2721|2223|2521|3001|2922|2323|2522|2121|2300|2622|2122|3000|2723|2702|3023|2400|2501),' train>>80_130.train
egrep ',[01],1410(3003|2900|2519|2621|2601|2619|2823|2119|2100|2800),' train>>80_130.valid


shuf data/site_train >data/r.site_train
wc -l data/r.site_train 
tail -5166566 data/r.site_train >data/cur.site_val
head -20666265 data/r.site_train >data/cur.site_train
wc -l data/cur.site*

shuf data/app_train >data/r.app_train
wc -l data/r.app_train 
tail -2919227 data/r.app_train >data/cur.app_val
head -11676911 data/r.app_train >data/cur.app_train
wc -l data/cur.app*

perl -ne 'print ((0 == $. % 5) ? $_ : "")'  somefile