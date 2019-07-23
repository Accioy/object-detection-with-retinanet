x={'P3';'P4';'P5';'P6';'P7'}
y=[10;222;82;1;0]
b=bar(y);
ch = get(b,'children');
set(gca,'XTickLabel',x);
