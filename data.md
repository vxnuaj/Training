### data notepad. 

We have 119 `.tex` Files.

<details><summary>Each `.tex` file has the respective count of characters:</summary>
53602 <br/>
515964 <br/>
1755966 <br/>
128257 <br/>
111861 <br/>
79077 <br/>
446143 <br/>
518046 <br/>
19148 <br/>
282487 <br/>
121106 <br/>
259505 <br/>
91790 <br/>
117313 <br/>
283408 <br/>
208667 <br/>
202679 <br/>
47325 <br/>
130357 <br/>
225563 <br/>
71486 <br/>
313092 <br/>
8517 <br/>
285173 <br/>
108310 <br/>
49354 <br/>
8116 <br/>
248506 <br/>
320339 <br/>
305631 <br/>
158874 <br/>
532916 <br/>
145962 <br/>
209788 <br/>
90539 <br/>
398496 <br/>
83969 <br/>
205146 <br/>
581058 <br/>
135779 <br/>
688548 <br/>
563748 <br/>
266274 <br/>
884130 <br/>
165432 <br/>
183880 <br/>
417436 <br/>
103862 <br/>
176068 <br/>
234988 <br/>
403439 <br/>
173284 <br/>
105196 <br/>
28347 <br/>
369450 <br/>
110552 <br/>
236880 <br/>
112376 <br/>
97400 <br/>
423101 <br/>
122896 <br/>
46629 <br/>
191161 <br/>
120415 <br/>
136944 <br/>
299062 <br/>
65680 <br/>
165 (this is empty bibliogrpahy, which will be kept in the training set)<br/> 
205089 <br/>
122421 <br/>
134809 <br/>
223470 <br/>
187208 <br/>
201811 <br/>
1474986 <br/>
107390 <br/>
211367 <br/>
147918 <br/>
444254 <br/>
6560 <br/>
172121 <br/>
173437 <br/>
5943 <br/>
254245 <br/>
1798 <br/>
168748 <br/>
192066 <br/>
202393 <br/>
232381 <br/>
80778 <br/>
221110 <br/>
123276 <br/>
262000 <br/>
337623 <br/>
166199 <br/>
62718 <br/>
185636 <br/>
370716 <br/>
11124 <br/>
130687 <br/>
165539 <br/>
252171 <br/>
95581 <br/>
202605 <br/>
212398 <br/>
52634 <br/>
40833 <br/>
352147 <br/>
913967 <br/>
311495 <br/>
21760 <br/>
222399 <br/>
131970 <br/>
184711 <br/>
350729 <br/>
234803 <br/>
132958 <br/>
42110 <br/>
222579 <br/>
</details>

no preprocessing done in terms of cleaning data. we're training  on raw `.tex` files such that the LSTM is able to accurately generate LaTeX that compiiles well.

### `data` folder

1. `char_to_idx.json` is a file that holds mappings from each character in the vocabulary to it's respective index.

2. `idx_to_char.json` is a file that holds mappings from each index to its respective character in the vocabulary

3. `numeric_sequences.pkl` is the file that holds the a python list of tuples, where within each tuple we have two lists which holds the input sequences and target sequences respectively.

    ```python
    import pickle as pkl

    with open('data/numeric_sequences.pkl', 'rb') as f:
        num_seq = pkl.load(f)

    type(num_seq) 
    type(num_seq[0]) 
    type(num_seq[0][1])
    ```

    ```
    > list ( the list holding all sets of input + target sequence pairs, length is equivalent to the number of train / test pairs. )
    > tuple ( the tuple holding the ith pair of input : target sequences, length is 2)
    > list ( the list holding the input or target, depending if you index [0] or [1] in the tuple. length is sequence length)
    ```

4. `tokenized_files.pkl` is the file that holds a python list of lists. the overarching list is length `119`, equivalent to the count of `.tex` files in the dataset -- as it holds all the `.tex` files in a tokenized format. the inner lists -- for the ith list, holds the tokenized version of the ith `.tex` file, tokenized character-wise.

5. `X_train.pt`, `y_train.pt`, `X_test.pt`, `y_test.pt` all hold the training and testing datasets (in numerical format, by index | SEE `char_to_idx.json` above !!), `X` being input sequences and `y` being target sequences, as pytorch tensors. These were generated as:

    ```python

    X, y = [], []
    n = .8 # train split size, relative to entire dataset.

    with open('data/numeric_sequences.pkl', 'rb') as f:
        try:
            while True:
                input_seq, target = pkl.load(f)
                X.append(input_seq)
                y.append(target)
        except EOFError:
            pass

    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    torch.save(X_train, 'data/X_train.pt')
    torch.save(y_train, 'data/y_train.pt')
    torch.save(X_test, 'data/X_test.pt')
    torch.save(y_test, 'data/y_test.pt')


    ```

