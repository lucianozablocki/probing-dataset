# Sequence mismatches: rnaglib (base) vs rnapdbee

7 mismatches.
Highlighted nucleotide(s) mark the diff.

---

### 1. `3k1v` chain `A`  —  deletion in rnapdbee

<pre>
rnaglib  : AGAGGUUCUAG<span style="background:#ffe066;font-weight:bold">C</span>CCCUCUAUAAAAAACUAA
rnapdbee : AGAGGUUCUAGCCCUCUAUAAAAAACUAA
dot-br   : (((((...[[[.)))))........]]].
</pre>

---

### 2. `5aox` chain `C`  —  deletion in rnapdbee

<pre>
rnaglib  : GCCGGGCGCGGUGGCUCACGCCUGUAAUCCCAGCACUUUGGGAGGCGAGGCGGGAGGAUCGCGAAC<span style="background:#ffe066;font-weight:bold">AC</span>GCGAGACCCCGUCUCUA
rnapdbee : GCCGGGCGCGGUGGCUCACGCCUGUAAUCCCAGCACUUUGGGAGGCGAGGCGGGAGGAUCGCGAACGCGAGACCCCGUCUCUA
dot-br   : ((((((((.(..[[[.).)))))....((((.]].]...)))))))((((((((.(..(((((..)))))..)))))))))..
</pre>

---

### 3. `5ns3` chain `C`  —  deletion in rnapdbee

<pre>
rnaglib  : CGCACCUGACCCCAUGCCGAACUCAGA<span style="background:#ffe066;font-weight:bold">A</span>GUGCG
rnapdbee : CGCACCUGACCCCAUGCCGAACUCAGAGUGCG
dot-br   : (((((((((...(.....)...)))).)))))
</pre>

---

### 4. `7d8o` chain `B`  —  deletion in rnapdbee

<pre>
rnaglib  : AUUUAGGUGAUUUGCUACCUUUAAGUGCAGCUAGAA<span style="background:#ffe066;font-weight:bold">A</span>
rnapdbee : AUUUAGGUGAUUUGCUACCUUUAAGUGCAGCUAGAA
dot-br   : ....((((....[[[.))))......]]].......
</pre>

---

### 5. `7mky` chain `A`  —  deletion in rnapdbee

<pre>
rnaglib  : CGGUGUAAGUGCAGCCCGUCUUACACCGUGCGGCACAGCGGAAACGCUGAUGUCGUA<span style="background:#ffe066;font-weight:bold">U</span>ACAGGGCU
rnapdbee : CGGUGUAAGUGCAGCCCGUCUUACACCGUGCGGCACAGCGGAAACGCUGAUGUCGUAACAGGGCU
dot-br   : (((((((((...[[[[[..)))))))))((((((((((((....))))).)))))))...]]]]]
</pre>

---

### 6. `8g9z` chain `E`  —  deletion in rnapdbee

<pre>
rnaglib  : GCCCGGAUGAUCCUCAGUGGUCUGGGGUGCAGGC<span style="background:#ffe066;font-weight:bold">U</span>AAACCUGUAGCUGUCUAGCGACAGAGUGGUUCAAUUCCACCUUUCGGGCGC<span style="background:#ffe066;font-weight:bold">C</span>
rnapdbee : GCCCGGAUGAUCCUCAGUGGUCUGGGGUGCAGGCAAACCUGUAGCUGUCUAGCGACAGAGUGGUUCAAUUCCACCUUUCGGGCGC
dot-br   : (((((((.(..((((((..[.))))))((((((....)))))).(((((....))))).((((..]....))))).)))))))..
</pre>

---

### 7. `8k1e` chain `B`  —  insertion in rnapdbee

<pre>
rnaglib  : GGCGCUGGUGGGGCACGUCCAGCGCU
rnapdbee : GGCGCUGGUGGGGCACGUCCAGCGCU<span style="background:#ffe066;font-weight:bold">GGGCCGGGGUUCGAGUCCCCGCAGUGUU</span>
dot-br   : ((((((((((..[)))((((((()))))))(((((..]....))))))))))))
</pre>
