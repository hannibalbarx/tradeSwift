BEGIN{
	FS=",";
	print "click, hour,C1,banner_pos,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21,8_f028772b,8_28905ebd,7_1fbe01fe,6_f3845767,6_7e091613,8_3e814130,7_e151e245,14_8a4875bd,8_50e219e0,6_7687a86e,14_d787e91b,6_98572c79,7_d9750ee7,14_1f0bc64f,7_5b08c53b,6_16a36ef3,7_5b4d2eda,7_856e6d3f,6_58a89a43,14_76dc4769"
}
{
	printf "%s,%s,%s,%s",$2,substr($3,7,2),$4,$5
	for (i=15;i<=23;i++) {printf ",%s",$i;}
	printf ",%d",$24
	if ($8=="f028772b") printf ",1"; else printf ",0";
	if ($8=="28905ebd") printf ",1"; else printf ",0";
	if ($7=="1fbe01fe") printf ",1"; else printf ",0";
	if ($6=="f3845767") printf ",1"; else printf ",0";
	if ($6=="7e091613") printf ",1"; else printf ",0";
	if ($8=="3e814130") printf ",1"; else printf ",0";
	if ($7=="e151e245") printf ",1"; else printf ",0";
	if ($14=="8a4875bd") printf ",1"; else printf ",0";
	if ($8=="50e219e0") printf ",1"; else printf ",0";
	if ($6=="7687a86e") printf ",1"; else printf ",0";
	if ($14=="d787e91b") printf ",1"; else printf ",0";
	if ($6=="98572c79") printf ",1"; else printf ",0";
	if ($7=="d9750ee7") printf ",1"; else printf ",0";
	if ($14=="1f0bc64f") printf ",1"; else printf ",0";
	if ($7=="5b08c53b") printf ",1"; else printf ",0";
	if ($6=="16a36ef3") printf ",1"; else printf ",0";
	if ($7=="5b4d2eda") printf ",1"; else printf ",0";
	if ($7=="856e6d3f") printf ",1"; else printf ",0";
	if ($6=="58a89a43") printf ",1"; else printf ",0";
	if ($14=="76dc4769") printf ",1"; else printf ",0";
	printf "\n"
}
