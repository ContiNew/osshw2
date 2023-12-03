import pandas as pd
import numpy as np

'''
OSS Project 2-1 
Author: 12183588 컴퓨터공학과 허대현
'''
def solve_q1(data_df:pd.DataFrame):
    #1.print the top 10 players in hits (안타, H), batting average (타율, avg), homerun (홈런, HR), 
    #and on-base percentage (출루율, OBP) for each year from 2015 to 2018.
    total_top10_df = data_df[['batter_name','H','avg','HR','OBP','year']]
    print("Solution for Question 1.")
    print("============================================================")

    for year in [2015,2016,2017,2018]:
        top10_of_year_df = total_top10_df[total_top10_df['year']==year] #각 연도에 해당되는 요소만 고른다.
        for ability in ['H','avg','HR','OBP']: #어빌리티 별로 반복
            target_df = (((top10_of_year_df.sort_values(by=ability,ascending=False))[['batter_name', ability]]).iloc[0:10,:]).to_numpy()
            #어빌리티를 기준으로 내림차 순으로 정렬하고, 여기서 주자 이름과, 어빌리티 열만 가져온 뒤에 상위 10개만 슬라이싱하고, nd_array 향테로 바꾼다. 
            print("Top 10 players with "+ ability+ " in "+ str(year)) # 해당 어빌리티로 해당 연도에 최고인 10명의 선수를 반복문으로 출력한다.
            rank = 1
            for player in target_df: # 0번째 인덱스에 batter_name, 1번 인덱스에 어빌리티에 해당하는 수치가 들어 있다.
                print(str(rank)+'. '+ player[0] + ' : '+ str(player[1]))
                rank = rank +1
            print()
    
    print("============================================================")
    print()
            

def solve_q2(data_df:pd.DataFrame):
    #2. Print the player with the highest war (승리 기여도) by position (cp) in 2018.
    data_2018_df = data_df[['batter_name','war','cp','year']] 
    data_2018_df = data_2018_df[data_2018_df['year'] == 2018]
    cp_info = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수'] 
    print("Solution for Question 2.")
    print("============================================================")
    for cp in cp_info: #포지션 별로 반복문 수행한다.
        target_df = data_2018_df[data_2018_df['cp']==cp].sort_values(by='war',ascending=False).iloc[0,:].to_numpy()
        #해당하는 포지션의 선수에 해당하는 행만 가져오고, war를 기준으로 내림차 정렬, 최상단의 행을 가져온다. 이를 ndarray 형태로 바꾼다
        print("Top player with war whose position is "+ cp + " in 2018")
        print(target_df[0] + ' : '+ str(target_df[1])) # 0번 인덱스에는 선수명, 2번째 인덱스에는 war 수치가 들어 있으므로 이를 이용해 출력한다.
        print()
    print("============================================================")
    print()

def solve_q3(data_df:pd.DataFrame):
    #3. Among R (득점), H (안타), HR (홈런), RBI (타점), SB (도루), war (승리 기여도), avg (타율), OBP (출루율), 
    # and SLG (장타율), which has the highest correlation with salary (연봉)? 
    print("Solution for Question 3.")
    print("============================================================")
    corr_df = data_df[['R','H','HR', 'RBI','SB','war','avg','OBP','SLG','salary']]
    returns = corr_df.corrwith(corr_df['salary']).sort_values(ascending=False).iloc[1:2].to_dict()
    #salary에 대한 상관계수를 구한뒤, 이를 내림차 순으로 정렬하여, 가장 큰 상관계수를 가지는 요소를 iloc 으로 추출한 뒤, 딕셔너리로 추출 
    returns = list(returns.items()) 
    # 딕셔너리 내장함수로 아이템을 추출 하고, 리스트형태로 바꾸어 인덱싱 가능하게 바꾼다.
    print(returns[0][0],"has the highest correlation with salary. corr : (" , returns[0][1], ")")
    print("============================================================")
    print()
    
    


if __name__=='__main__':
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
    # csv 파일을 데이터 프레임의 형태로 가져온다.
    solve_q1(data_df)
    solve_q2(data_df)
    solve_q3(data_df)
    
    