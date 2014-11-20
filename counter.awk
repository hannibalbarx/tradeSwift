BEGIN{
	FS=","
	cur_hr=""

	count=0
	count_site=0
	count_app=0
	
	count_click=0
	count_site_click=0
	count_app_click=0
	
}
{
	if ($3!=cur_hr){
		print cur_hr, count, count_site, count_app, count_click, count_site_click, count_app_click
		count=0
		count_site=0
		count_app=0
		
		count_click=0
		count_site_click=0
		count_app_click=0
		cur_hr=$3
	}
	count++
	if ($2=="1") count_click++;
	if ($9=="ecad2386" && $10=="7801e8d9" && $11=="07d7df22"){
		count_site++
		if ($2=="1") count_site_click++;
	}
	if ($6=="85f751fd" && $7=="c4e18dd6" && $8=="50e219e0"){
		count_app++
		if ($2=="1") count_app_click++;
	}
}
END{
	print cur_hr, count, count_site, count_app, count_click, count_site_click, count_app_click
}