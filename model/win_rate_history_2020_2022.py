"""
Real historical ESPN win-rate team data for 2020-2022 -- gathered
directly from ESPN's real published season-recap articles (see
model/win_rate_metrics.py for the general module; these specific
years use the older list-based article format, not the newer table
format, so they're parsed from raw text captured directly off the
live pages, not re-fetched).

Real URLs (verified against espnscrapeR's own hardcoded history):
  2020: espn.com/nfl/story/_/id/29939464/...
  2021: espn.com/nfl/story/_/id/32176833/...
  2022: espn.com/nfl/story/_/id/34536376/...
"""

import re

TEAM_NAME_TO_ABBR = {
    "Los Angeles Rams": "LA", "Pittsburgh Steelers": "PIT", "Buffalo Bills": "BUF",
    "Washington Football Team": "WAS", "Washington Commanders": "WAS",
    "Atlanta Falcons": "ATL", "Tampa Bay Buccaneers": "TB", "New York Jets": "NYJ",
    "Baltimore Ravens": "BAL", "Philadelphia Eagles": "PHI", "Houston Texans": "HOU",
    "Jacksonville Jaguars": "JAX", "Carolina Panthers": "CAR", "Arizona Cardinals": "ARI",
    "Cleveland Browns": "CLE", "Indianapolis Colts": "IND", "Las Vegas Raiders": "LV",
    "Tennessee Titans": "TEN", "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC",
    "Green Bay Packers": "GB", "Miami Dolphins": "MIA", "New Orleans Saints": "NO",
    "Seattle Seahawks": "SEA", "Denver Broncos": "DEN", "Chicago Bears": "CHI",
    "San Francisco 49ers": "SF", "Detroit Lions": "DET", "Dallas Cowboys": "DAL",
    "Cincinnati Bengals": "CIN", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New York Giants": "NYG",
}


def parse_ranked_list(text):
    """Parses '1. Team Name, 54%' formatted lines into {abbr: pct}."""
    result = {}
    for line in text.strip().split("\n"):
        match = re.match(r"\d+\.\s+(.+?),\s+(\d+)%", line.strip())
        if match:
            team_name, pct = match.groups()
            abbr = TEAM_NAME_TO_ABBR.get(team_name.strip())
            if abbr:
                result[abbr] = int(pct)
    return result


PRWR_2020 = parse_ranked_list("""
1. Pittsburgh Steelers, 54%
2. Buffalo Bills, 52%
3. Washington Football Team, 50%
4. Atlanta Falcons, 49%
5. Tampa Bay Buccaneers, 49%
6. New York Jets, 48%
7. Baltimore Ravens, 48%
8. Philadelphia Eagles, 48%
9. Los Angeles Rams, 47%
10. Houston Texans, 45%
11. Jacksonville Jaguars, 45%
12. Carolina Panthers, 44%
13. Arizona Cardinals, 44%
14. Cleveland Browns, 44%
15. Indianapolis Colts, 42%
16. Las Vegas Raiders, 42%
17. Tennessee Titans, 42%
18. Kansas City Chiefs, 42%
19. Los Angeles Chargers, 42%
20. Green Bay Packers, 41%
21. Miami Dolphins, 40%
22. New Orleans Saints, 40%
23. Seattle Seahawks, 40%
24. Denver Broncos, 39%
25. Chicago Bears, 38%
26. San Francisco 49ers, 38%
27. Detroit Lions, 38%
28. Dallas Cowboys, 37%
29. Cincinnati Bengals, 37%
30. Minnesota Vikings, 36%
31. New England Patriots, 36%
32. New York Giants, 31%
""")

RSWR_2020 = parse_ranked_list("""
1. New York Jets, 33%
2. Tampa Bay Buccaneers, 32%
3. Las Vegas Raiders, 32%
4. Atlanta Falcons, 32%
5. Philadelphia Eagles, 31%
6. New England Patriots, 31%
7. Indianapolis Colts, 31%
8. Tennessee Titans, 31%
9. Baltimore Ravens, 31%
10. Washington Football Team, 30%
11. Houston Texans, 30%
12. San Francisco 49ers, 30%
13. New York Giants, 30%
14. Los Angeles Chargers, 30%
15. Miami Dolphins, 30%
16. Los Angeles Rams, 30%
17. Chicago Bears, 30%
18. Jacksonville Jaguars, 30%
19. New Orleans Saints, 30%
20. Seattle Seahawks, 30%
21. Pittsburgh Steelers, 30%
22. Arizona Cardinals, 29%
23. Buffalo Bills, 29%
24. Green Bay Packers, 29%
25. Detroit Lions, 29%
26. Kansas City Chiefs, 29%
27. Cincinnati Bengals, 28%
28. Carolina Panthers, 28%
29. Denver Broncos, 28%
30. Cleveland Browns, 26%
31. Dallas Cowboys, 25%
32. Minnesota Vikings, 24%
""")

PBWR_2020 = parse_ranked_list("""
1. Green Bay Packers, 74%
2. Cleveland Browns, 71%
3. Arizona Cardinals, 67%
4. Buffalo Bills, 64%
5. New Orleans Saints, 63%
6. Kansas City Chiefs, 63%
7. Los Angeles Rams, 63%
8. Baltimore Ravens, 62%
9. Seattle Seahawks, 62%
10. Las Vegas Raiders, 60%
11. Philadelphia Eagles, 60%
12. Indianapolis Colts, 60%
13. New England Patriots, 59%
14. Washington Football Team, 59%
15. Chicago Bears, 58%
16. Atlanta Falcons, 57%
17. Tampa Bay Buccaneers, 57%
18. Minnesota Vikings, 56%
19. Houston Texans, 56%
20. Detroit Lions, 55%
21. Denver Broncos, 54%
22. San Francisco 49ers, 54%
23. Carolina Panthers, 53%
24. Tennessee Titans, 53%
25. Jacksonville Jaguars, 51%
26. Dallas Cowboys, 51%
27. Miami Dolphins, 51%
28. Pittsburgh Steelers, 51%
29. Cincinnati Bengals, 50%
29. New York Jets, 50%
31. Los Angeles Chargers, 47%
32. New York Giants, 46%
""")

RBWR_2020 = parse_ranked_list("""
1. Green Bay Packers, 74%
2. Philadelphia Eagles, 73%
3. Washington Football Team, 73%
4. Baltimore Ravens, 73%
5. Houston Texans, 73%
6. Arizona Cardinals, 72%
7. Carolina Panthers, 72%
8. Indianapolis Colts, 72%
9. New Orleans Saints, 72%
10. New England Patriots, 72%
11. Cincinnati Bengals, 71%
12. Chicago Bears, 71%
13. Cleveland Browns, 71%
14. Detroit Lions, 70%
15. Dallas Cowboys, 70%
16. Seattle Seahawks, 70%
17. Tampa Bay Buccaneers, 70%
18. New York Giants, 70%
19. Los Angeles Rams, 70%
20. Tennessee Titans, 70%
21. Minnesota Vikings, 70%
22. Denver Broncos, 70%
23. Miami Dolphins, 69%
24. Pittsburgh Steelers, 69%
25. San Francisco 49ers, 69%
26. Jacksonville Jaguars, 69%
27. Las Vegas Raiders, 69%
28. Atlanta Falcons, 69%
29. Buffalo Bills, 69%
30. New York Jets, 67%
31. Kansas City Chiefs, 67%
32. Los Angeles Chargers, 67%
""")

PRWR_2021 = parse_ranked_list("""
1. Los Angeles Rams, 53%
2. Carolina Panthers, 51%
3. Cleveland Browns, 50%
4. Philadelphia Eagles, 49%
5. San Francisco 49ers, 46%
6. Buffalo Bills, 46%
7. Kansas City Chiefs, 44%
8. Las Vegas Raiders, 44%
9. Miami Dolphins, 43%
10. Tampa Bay Buccaneers, 43%
11. Los Angeles Chargers, 42%
12. Dallas Cowboys, 42%
13. Arizona Cardinals, 41%
14. Baltimore Ravens, 41%
15. Pittsburgh Steelers, 41%
16. New York Jets, 41%
17. Chicago Bears, 41%
18. Houston Texans, 40%
19. Washington Football Team, 40%
20. Seattle Seahawks, 39%
21. Tennessee Titans, 39%
22. Jacksonville Jaguars, 39%
23. New England Patriots, 37%
24. Indianapolis Colts, 36%
25. Cincinnati Bengals, 36%
26. New Orleans Saints, 36%
27. Green Bay Packers, 35%
28. Atlanta Falcons, 34%
29. Minnesota Vikings, 34%
30. New York Giants, 34%
31. Detroit Lions, 33%
32. Denver Broncos, 31%
""")

RSWR_2021 = parse_ranked_list("""
1. Los Angeles Rams, 35%
2. Baltimore Ravens, 33%
3. San Francisco 49ers, 33%
4. Tennessee Titans, 33%
5. New England Patriots, 33%
6. New York Jets, 32%
7. Seattle Seahawks, 32%
8. New Orleans Saints, 32%
9. Las Vegas Raiders, 32%
10. Miami Dolphins, 31%
11. Carolina Panthers, 31%
12. Indianapolis Colts, 31%
13. Washington Football Team, 31%
14. Dallas Cowboys, 31%
15. Houston Texans, 31%
16. Buffalo Bills, 31%
17. Philadelphia Eagles, 31%
18. Green Bay Packers, 30%
19. Tampa Bay Buccaneers, 30%
20. Cleveland Browns, 30%
21. Los Angeles Chargers, 30%
22. Arizona Cardinals, 29%
23. Detroit Lions, 29%
24. New York Giants, 29%
25. Cincinnati Bengals, 29%
26. Atlanta Falcons, 28%
27. Denver Broncos, 28%
28. Jacksonville Jaguars, 28%
29. Pittsburgh Steelers, 28%
30. Chicago Bears, 27%
31. Minnesota Vikings, 27%
32. Kansas City Chiefs, 27%
""")

PBWR_2021 = parse_ranked_list("""
1. Los Angeles Rams, 68%
2. Kansas City Chiefs, 68%
3. Philadelphia Eagles, 67%
4. Cleveland Browns, 67%
5. Green Bay Packers, 66%
6. Chicago Bears, 66%
7. New Orleans Saints, 66%
8. Buffalo Bills, 64%
9. Washington Football Team, 63%
10. Baltimore Ravens, 62%
11. New England Patriots, 62%
12. Arizona Cardinals, 61%
13. Los Angeles Chargers, 61%
14. San Francisco 49ers, 61%
15. Seattle Seahawks, 61%
16. Denver Broncos, 61%
17. New York Jets, 61%
18. Jacksonville Jaguars, 60%
19. Tampa Bay Buccaneers, 60%
20. Indianapolis Colts, 60%
21. Las Vegas Raiders, 59%
22. Detroit Lions, 58%
23. Dallas Cowboys, 58%
24. Tennessee Titans, 56%
25. Minnesota Vikings, 54%
26. Atlanta Falcons, 54%
27. Houston Texans, 54%
28. New York Giants, 54%
29. Carolina Panthers, 50%
30. Cincinnati Bengals, 49%
31. Pittsburgh Steelers, 49%
32. Miami Dolphins, 47%
""")

RBWR_2021 = parse_ranked_list("""
1. Washington Football Team, 75%
2. Philadelphia Eagles, 74%
3. Kansas City Chiefs, 74%
4. Green Bay Packers, 73%
5. Baltimore Ravens, 73%
6. Dallas Cowboys, 73%
7. Indianapolis Colts, 72%
8. Cleveland Browns, 72%
9. Miami Dolphins, 71%
10. Cincinnati Bengals, 71%
11. Chicago Bears, 71%
12. Los Angeles Rams, 71%
13. Minnesota Vikings, 71%
14. New York Giants, 71%
15. Denver Broncos, 70%
16. New England Patriots, 70%
17. Detroit Lions, 70%
18. San Francisco 49ers, 70%
19. Arizona Cardinals, 70%
20. Los Angeles Chargers, 70%
21. Jacksonville Jaguars, 70%
22. Tampa Bay Buccaneers, 70%
23. Buffalo Bills, 69%
24. Tennessee Titans, 69%
25. New Orleans Saints, 69%
26. Carolina Panthers, 68%
27. New York Jets, 68%
28. Seattle Seahawks, 68%
29. Atlanta Falcons, 67%
30. Pittsburgh Steelers, 67%
31. Las Vegas Raiders, 67%
32. Houston Texans, 65%
""")

PRWR_2022 = parse_ranked_list("""
1. Philadelphia Eagles, 52%
2. Dallas Cowboys, 52%
3. Miami Dolphins, 50%
4. Arizona Cardinals, 47%
5. San Francisco 49ers, 46%
6. New York Giants, 45%
7. Los Angeles Rams, 45%
8. Jacksonville Jaguars, 44%
9. Denver Broncos, 44%
10. New York Jets, 44%
11. Buffalo Bills, 43%
12. Green Bay Packers, 43%
13. Cleveland Browns, 43%
14. Houston Texans, 43%
15. Kansas City Chiefs, 41%
16. Washington Commanders, 41%
17. Baltimore Ravens, 40%
18. Pittsburgh Steelers, 40%
19. Carolina Panthers, 39%
20. Tennessee Titans, 38%
21. Cincinnati Bengals, 37%
22. Tampa Bay Buccaneers, 37%
23. Chicago Bears, 37%
24. Las Vegas Raiders, 37%
25. Detroit Lions, 36%
26. Atlanta Falcons, 35%
27. Minnesota Vikings, 34%
28. Seattle Seahawks, 34%
29. New England Patriots, 33%
30. Los Angeles Chargers, 33%
31. Indianapolis Colts, 33%
32. New Orleans Saints, 29%
""")

RSWR_2022 = parse_ranked_list("""
1. Tennessee Titans, 36%
2. New York Jets, 33%
3. Los Angeles Rams, 33%
4. Indianapolis Colts, 32%
5. Carolina Panthers, 32%
6. San Francisco 49ers, 32%
7. Buffalo Bills, 32%
8. Miami Dolphins, 32%
9. Washington Commanders, 32%
10. Denver Broncos, 31%
11. Cleveland Browns, 31%
12. New England Patriots, 31%
13. Baltimore Ravens, 31%
14. Philadelphia Eagles, 31%
15. Chicago Bears, 30%
16. Tampa Bay Buccaneers, 30%
17. Houston Texans, 30%
18. Minnesota Vikings, 30%
19. Jacksonville Jaguars, 30%
20. Seattle Seahawks, 30%
21. Arizona Cardinals, 30%
22. Las Vegas Raiders, 30%
23. Los Angeles Chargers, 29%
24. Dallas Cowboys, 29%
25. New Orleans Saints, 28%
26. Detroit Lions, 28%
27. Pittsburgh Steelers, 28%
28. New York Giants, 27%
29. Atlanta Falcons, 27%
30. Cincinnati Bengals, 27%
31. Green Bay Packers, 27%
32. Kansas City Chiefs, 26%
""")

PBWR_2022 = parse_ranked_list("""
1. Kansas City Chiefs, 75%
2. Chicago Bears, 68%
3. Cleveland Browns, 68%
4. Buffalo Bills, 67%
5. Green Bay Packers, 66%
6. Baltimore Ravens, 66%
7. Pittsburgh Steelers, 65%
8. Seattle Seahawks, 63%
9. Denver Broncos, 62%
10. Las Vegas Raiders, 62%
11. Carolina Panthers, 62%
12. Philadelphia Eagles, 62%
13. Los Angeles Rams, 61%
14. Arizona Cardinals, 61%
15. New England Patriots, 61%
16. New Orleans Saints, 60%
17. Houston Texans, 60%
18. Detroit Lions, 60%
19. Atlanta Falcons, 59%
20. San Francisco 49ers, 59%
21. New York Jets, 57%
22. Minnesota Vikings, 57%
23. Los Angeles Chargers, 57%
24. Miami Dolphins, 55%
25. Tampa Bay Buccaneers, 55%
26. Tennessee Titans, 54%
27. Washington Commanders, 53%
28. Dallas Cowboys, 53%
29. New York Giants, 52%
30. Cincinnati Bengals, 50%
31. Jacksonville Jaguars, 49%
32. Indianapolis Colts, 49%
""")

RBWR_2022 = parse_ranked_list("""
1. Baltimore Ravens, 77%
2. Philadelphia Eagles, 75%
3. Kansas City Chiefs, 74%
4. Denver Broncos, 74%
5. Chicago Bears, 74%
6. Arizona Cardinals, 73%
7. Dallas Cowboys, 73%
8. Green Bay Packers, 72%
9. Detroit Lions, 72%
10. Cincinnati Bengals, 72%
11. Las Vegas Raiders, 72%
12. Cleveland Browns, 72%
13. Los Angeles Rams, 72%
14. Pittsburgh Steelers, 72%
15. Carolina Panthers, 72%
16. Tennessee Titans, 71%
17. San Francisco 49ers, 71%
18. Minnesota Vikings, 71%
19. Washington Commanders, 71%
20. New Orleans Saints, 71%
21. Miami Dolphins, 71%
22. Buffalo Bills, 71%
23. Indianapolis Colts, 71%
24. Seattle Seahawks, 71%
25. Atlanta Falcons, 71%
26. New York Giants, 70%
27. Houston Texans, 70%
28. Los Angeles Chargers, 70%
29. Jacksonville Jaguars, 70%
30. New York Jets, 69%
31. Tampa Bay Buccaneers, 69%
32. New England Patriots, 68%
""")

WIN_RATES_BY_SEASON = {
    2020: {"prwr": PRWR_2020, "rswr": RSWR_2020, "pbwr": PBWR_2020, "rbwr": RBWR_2020},
    2021: {"prwr": PRWR_2021, "rswr": RSWR_2021, "pbwr": PBWR_2021, "rbwr": RBWR_2021},
    2022: {"prwr": PRWR_2022, "rswr": RSWR_2022, "pbwr": PBWR_2022, "rbwr": RBWR_2022},
}

if __name__ == "__main__":
    for season, metrics in WIN_RATES_BY_SEASON.items():
        for metric_name, teams in metrics.items():
            print(f"{season} {metric_name}: {len(teams)} teams")
            assert len(teams) == 32, f"Expected 32 teams for {season} {metric_name}, got {len(teams)}"
    print("\nPASS: all 12 metric-seasons have complete 32-team coverage")
