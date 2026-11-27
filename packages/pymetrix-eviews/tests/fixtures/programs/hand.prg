' A made-up program for pymetrix-eviews's tests (pymetrix's own work, BSD-3-Clause): every value it makes is
' worked out by hand in test_program.py.
wfcreate(wf=hand, page=q) q 2000q1 2001q4
series t = 1
smpl 2000q2 @last
t = t(-1) + 1
smpl @all
series y = 2*t + 1
scalar half = 0.5
!two = 2
series z = y*half - !two   ' (2t + 1)/2 - 2
smpl 2000q3 2001q2
series v = 10
series w
smpl if y > 9
scalar m = @mean(y)
scalar sd = @stdev(y)
smpl @first 2000q4
series c = @pch(y)
smpl @all
series dz = d(z)
frml f = y + m
group g t y
t.label(u) quarters
y.displayname Why
rename v vv
delete w
vector(2) b
b.fill 1, 2
matrix(2,3) mm
mm.fill(b=r) 1, 2, 3, _
  4, 5, 6
string s = "abc"
%x = s + "d"
series q{%x} = y + _
  1
graph gr.line y z
equation eq.ls y c t
series dn = @date
for !i = 1 to 2
  series a{!i} = !i
next
wfsave hand
