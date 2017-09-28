# for x in models/InceptionV3_*; do grep "nmt.newtrain - INFO - BLEU =" $x/training.log | cut -d ' ' -f 10 | sort -rh | head -n1; done | perl tools/stats.pl

use List::Util qw(max min sum); @a=();while(<>){$sqsum+=$_*$_; push(@a,$_)}; $n=@a;$s=sum(@a);$a=$s/@a;$m=max(@a);$mm=min(@a);$std=sqrt($sqsum/$n-($s/$n)*($s/$n));$mid=int @a/2;@srtd=sort @a;if(@a%2){$med=$srtd[$mid];}else{$med=($srtd[$mid-1]+$srtd[$mid])/2;};print "records:$n\nsum:$s\navg:$a\nstd:$std\nmed:$med\max:$m\min:$mm";
