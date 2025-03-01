# Fantasy Baseball Draft Tool

A Python-based tool for fantasy baseball draft analysis and player forecasting.

## Overview

This tool helps fantasy baseball players make informed draft decisions by:
- Processing historical baseball data from Retrosheet
- Analyzing player performance across relevant fantasy categories
- Forecasting player performance for the upcoming season
- Ranking players based on projected value in H2H categories leagues
- Providing a command-line interface for draft assistance

## League Settings

- **Format**: Head-to-Head Categories
- **Hitting Categories (6)**: HR, OBP, R, RBI, SB, TB
- **Pitching Categories (5)**: ERA, WHIP, K, Saves+Holds, Wins+Quality Starts

## Project Structure

```
fantasy-baseball-draft-tool/
├── data/                      # Directory for data files
│   ├── raw/                   # Raw Retrosheet data
│   ├── processed/             # Processed player statistics
│   └── projections/           # Generated player projections
├── src/                       # Source code
│   ├── data_processing/       # Data processing modules
│   ├── analysis/              # Statistical analysis modules
│   ├── forecasting/           # Forecasting model modules
│   ├── ranking/               # Player ranking modules
│   └── draft_tool/            # Draft interface modules
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Test modules
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download Retrosheet data to the `data/raw/` directory
4. Run data processing: `python src/data_processing/process_retrosheet.py`
5. Generate projections: `python src/forecasting/generate_projections.py`
6. Start draft tool: `python src/draft_tool/cli.py`

## Data Sources

This tool uses data from [Retrosheet](https://www.retrosheet.org/), which provides play-by-play data for MLB games.

Here are the files and their headers and first row

allplayers.csv - 
```
id,last,first,bat,throw,team,g,g_p,g_sp,g_rp,g_c,g_1b,g_2b,g_3b,g_ss,g_lf,g_cf,g_rf,g_of,g_dh,g_ph,g_pr,first_g,last_g,season
andej101,Anderson,John,B,R,MLA,138,0,0,0,0,125,0,0,0,12,0,1,13,0,0,0,19010425,19010928,1901
```

batting.csv - 
```
gid,id,team,b_lp,b_seq,stattype,b_pa,b_ab,b_r,b_h,b_d,b_t,b_hr,b_rbi,b_sh,b_sf,b_hbp,b_w,b_iw,b_k,b_sb,b_cs,b_gdp,b_xi,b_roe,dh,ph,pr,date,number,site,vishome,opp,win,loss,tie,gametype,box,pbp
PHI190104180,davil101,BRO,1,1,value,5,5,1,1,0,0,0,0,0,,0,0,,0,0,,,0,,,,,19010418,,PHI09,v,PHI,1,0,0,regular,y,
```

fielding.csv - 
```
gid,id,team,d_seq,d_pos,stattype,d_ifouts,d_po,d_a,d_e,d_dp,d_tp,d_pb,d_wp,d_sb,d_cs,d_gs,date,number,site,vishome,opp,win,loss,tie,gametype,box,pbp
PHI190104180,davil101,BRO,1,7,value,27,3,0,1,0,0,,,,,1,19010418,,PHI09,v,PHI,1,0,0,regular,y,
```

pitching.csv -
```
gid,id,team,p_seq,stattype,p_ipouts,p_noout,p_bfp,p_h,p_d,p_t,p_hr,p_r,p_er,p_w,p_iw,p_k,p_hbp,p_wp,p_bk,p_sh,p_sf,p_sb,p_cs,p_pb,wp,lp,save,p_gs,p_gf,p_cg,date,number,site,vishome,opp,win,loss,tie,gametype,box,pbp
PHI190104180,donob101,BRO,1,value,27,,45,12,2,1,0,7,4,5,,4,0,0,0,0,,,,,1,,,1,,1,19010418,,PHI09,v,PHI,1,0,0,regular,y,
```

plays.csv - 
```
gid,event,inning,top_bot,vis_home,ballpark,batteam,pitteam,batter,pitcher,lp,bat_f,bathand,pithand,count,pitches,nump,pa,ab,single,double,triple,hr,sh,sf,hbp,walk,iw,k,xi,oth,othout,noout,bip,bunt,ground,fly,line,gdp,othdp,tp,wp,pb,bk,oa,di,sb2,sb3,sbh,cs2,cs3,csh,pko1,pko2,pko3,k_safe,e1,e2,e3,e4,e5,e6,e7,e8,e9,outs_pre,outs_post,br1_pre,br2_pre,br3_pre,br1_post,br2_post,br3_post,run_b,run1,run2,run3,prun1,prun2,prun3,runs,rbi,er,tur,f2,f3,f4,f5,f6,f7,f8,f9,po0,po1,po2,po3,po4,po5,po6,po7,po8,po9,a1,a2,a3,a4,a5,a6,a7,a8,a9,batout1,batout2,batout3,brout_b,brout1,brout2,brout3,firstf,loc,hittype,dpopp,pivot,pn,date,gametype,pbp
CUX190309140,D#,1,0,0,TRE02,PHG,CUX,pattj102,fostr105,1,7,?,R,??,,,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,,,,,pattj102,,,,,,,,,0,0,0,0,willc112,jordb102,granc101,hillj106,johng103,jackw103,payna101,smitb115,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,,,0,0,1,19030914,championship,deduced
```

gameinfo.csv - 
```
gid,visteam,hometeam,site,date,number,starttime,daynight,innings,tiebreaker,usedh,htbf,timeofgame,attendance,fieldcond,precip,sky,temp,winddir,windspeed,oscorer,forfeit,suspend,umphome,ump1b,ump2b,ump3b,umplf,umprf,wp,lp,save,gametype,vruns,hruns,wteam,lteam,line,batteries,lineups,box,pbp,season
PHI190104180,BRO,PHI,PHI09,19010418,,0:00PM,day,,,false,,125,4593,unknown,unknown,unknown,0,unknown,-1,,,,colgh901,(none),(none),(none),,,donob101,dunnj102,,regular,12,7,BRO,PHI,y,both,y,y,,1901
BRO190104190,PHI,BRO,NYC12,19010419,,0:00PM,day,,,false,,116,7600,unknown,unknown,unknown,0,unknown,-1,,,,colgh901,(none),(none),(none),,,mccag101,townh101,,regular,2,10,BRO,PHI,y,both,y,y,,1901
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Retrosheet for providing the historical baseball data
