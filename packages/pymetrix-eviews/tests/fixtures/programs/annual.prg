' A second made-up program (pymetrix's own work, BSD-3-Clause): two pages, an annual one made with pagecreate,
' samples by year and with offsets, a named sample, and a series filled by hand.
wfcreate(page=a) a 1990 1999
series x = 1
smpl 1991 @last
x = x(-1)*2
smpl @first+2 @last-2
series y = x - 1
sample mid 1993 1996
smpl mid
series k = 5
smpl @all
x.fill(o=1998) 7, na
pagecreate(page=q) q 1990q1 1990q4
series u = 3
pageselect a
series v = 1 + 0
