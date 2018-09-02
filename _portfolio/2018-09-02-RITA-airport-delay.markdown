---
title:  "The most common causes of flight delays?"
date:   2018-09-02
tags: [Flight_delays]

header:
  image: "/images/RITA-airport-delay/airport_cover.jpg"

excerpt: "Flight Delays, Visualization, Tableau"
---

In India we know the flights were most delay because we didn't have that many airports and airplanes but I was curious in USA what are the main common causes of flight delays.
So I collected the data from [RITA](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) and then use [Tableau](https://www.tableau.com/) to see insights from the data.

## Common Causes
Below, we can see that Late aircraft causes, on average, a 20 minute delay. Followed by delays of around 17 minutes from the NAS (air traffic control) and over 15 minutes by the carrier (airline).

![](/images/RITA-airport-delay/1.png?raw=true)


## Delays throughout the year
My first expectations was that in colder months the delay would be maximus but the plot shows that the delays increase in June, July and December because the resources are under more strain. 
Interestingly enough, weather delay is fairly low in colder months. Perhaps, aeronautical engineering has improved over the years and airlines can cope better with adverse weather conditions.

![](/images/RITA-airport-delay/2.png?raw=true)


## Delays by Airport
From the graphic on the left hand side of the plot, we can see that Marquette has an average delay of over an hour and a half! However, this seems to be mainly due to the late arrivals of aircraft.
It's also worrying to find that popular airports Chicago O'Hare, JFK, Newark still have delays of over an hour. This seems to be due to NAS delays - probably due to air traffic in the area.

![](/images/RITA-airport-delay/3.png?raw=true)

***

For those interested, the Tableau Worksheet, Dashboard and the Story to create this chart from the RITA [data](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) can be found in the [Tableau Public](https://public.tableau.com/profile/souvik.banerjee#!/vizhome/RITAFlightData_0/RITA_delays?publish=yes) for this post.


