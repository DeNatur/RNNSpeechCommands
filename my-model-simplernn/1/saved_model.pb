Þ
Ï£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.42v2.3.3-137-gea90cf44f738

mel_stft/real_kernelsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namemel_stft/real_kernels

)mel_stft/real_kernels/Read/ReadVariableOpReadVariableOpmel_stft/real_kernels*(
_output_shapes
:*
dtype0

mel_stft/imag_kernelsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namemel_stft/imag_kernels

)mel_stft/imag_kernels/Read/ReadVariableOpReadVariableOpmel_stft/imag_kernels*(
_output_shapes
:*
dtype0

mel_stft/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P*"
shared_namemel_stft/Variable
x
%mel_stft/Variable/Read/ReadVariableOpReadVariableOpmel_stft/Variable*
_output_shapes
:	P*
dtype0
{
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(@* 
shared_namedense_37/kernel
t
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes
:	(@*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:@*
dtype0
z
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_38/kernel
s
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes

:@ *
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
: *
dtype0
z
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: $* 
shared_namedense_39/kernel
s
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes

: $*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
:$*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
ª
'simple_rnn_13/simple_rnn_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}@*8
shared_name)'simple_rnn_13/simple_rnn_cell_13/kernel
£
;simple_rnn_13/simple_rnn_cell_13/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_13/simple_rnn_cell_13/kernel*
_output_shapes

:}@*
dtype0
¾
1simple_rnn_13/simple_rnn_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*B
shared_name31simple_rnn_13/simple_rnn_cell_13/recurrent_kernel
·
Esimple_rnn_13/simple_rnn_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_13/simple_rnn_cell_13/recurrent_kernel*
_output_shapes

:@@*
dtype0
¢
%simple_rnn_13/simple_rnn_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%simple_rnn_13/simple_rnn_cell_13/bias

9simple_rnn_13/simple_rnn_cell_13/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_13/simple_rnn_cell_13/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/mel_stft/real_kernels/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/mel_stft/real_kernels/m

0Adam/mel_stft/real_kernels/m/Read/ReadVariableOpReadVariableOpAdam/mel_stft/real_kernels/m*(
_output_shapes
:*
dtype0

Adam/mel_stft/imag_kernels/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/mel_stft/imag_kernels/m

0Adam/mel_stft/imag_kernels/m/Read/ReadVariableOpReadVariableOpAdam/mel_stft/imag_kernels/m*(
_output_shapes
:*
dtype0

Adam/mel_stft/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P*)
shared_nameAdam/mel_stft/Variable/m

,Adam/mel_stft/Variable/m/Read/ReadVariableOpReadVariableOpAdam/mel_stft/Variable/m*
_output_shapes
:	P*
dtype0

Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(@*'
shared_nameAdam/dense_37/kernel/m

*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes
:	(@*
dtype0

Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_38/kernel/m

*Adam/dense_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/m*
_output_shapes

:@ *
dtype0

Adam/dense_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_38/bias/m
y
(Adam/dense_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/m*
_output_shapes
: *
dtype0

Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: $*'
shared_nameAdam/dense_39/kernel/m

*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes

: $*
dtype0

Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/dense_39/bias/m
y
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes
:$*
dtype0
¸
.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}@*?
shared_name0.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m
±
BAdam/simple_rnn_13/simple_rnn_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m*
_output_shapes

:}@*
dtype0
Ì
8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*I
shared_name:8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m
Å
LAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
°
,Adam/simple_rnn_13/simple_rnn_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/simple_rnn_13/simple_rnn_cell_13/bias/m
©
@Adam/simple_rnn_13/simple_rnn_cell_13/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_13/simple_rnn_cell_13/bias/m*
_output_shapes
:@*
dtype0

Adam/mel_stft/real_kernels/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/mel_stft/real_kernels/v

0Adam/mel_stft/real_kernels/v/Read/ReadVariableOpReadVariableOpAdam/mel_stft/real_kernels/v*(
_output_shapes
:*
dtype0

Adam/mel_stft/imag_kernels/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/mel_stft/imag_kernels/v

0Adam/mel_stft/imag_kernels/v/Read/ReadVariableOpReadVariableOpAdam/mel_stft/imag_kernels/v*(
_output_shapes
:*
dtype0

Adam/mel_stft/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P*)
shared_nameAdam/mel_stft/Variable/v

,Adam/mel_stft/Variable/v/Read/ReadVariableOpReadVariableOpAdam/mel_stft/Variable/v*
_output_shapes
:	P*
dtype0

Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(@*'
shared_nameAdam/dense_37/kernel/v

*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes
:	(@*
dtype0

Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_38/kernel/v

*Adam/dense_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/v*
_output_shapes

:@ *
dtype0

Adam/dense_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_38/bias/v
y
(Adam/dense_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/v*
_output_shapes
: *
dtype0

Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: $*'
shared_nameAdam/dense_39/kernel/v

*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes

: $*
dtype0

Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/dense_39/bias/v
y
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes
:$*
dtype0
¸
.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}@*?
shared_name0.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v
±
BAdam/simple_rnn_13/simple_rnn_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v*
_output_shapes

:}@*
dtype0
Ì
8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*I
shared_name:8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v
Å
LAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
°
,Adam/simple_rnn_13/simple_rnn_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v
©
@Adam/simple_rnn_13/simple_rnn_cell_13/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
ÑO
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*O
valueOBÿN BøN
ò
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api

signatures
{
_inbound_nodes
_outbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
µ
_inbound_nodes
dft_real_kernels
dft_imag_kernels
freq2mel
_outbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
{
 _inbound_nodes
!_outbound_nodes
"regularization_losses
#trainable_variables
$	variables
%	keras_api
{
&_inbound_nodes
'_outbound_nodes
(regularization_losses
)trainable_variables
*	variables
+	keras_api

,cell
-_inbound_nodes
.
state_spec
/_outbound_nodes
0regularization_losses
1trainable_variables
2	variables
3	keras_api
{
4_inbound_nodes
5_outbound_nodes
6regularization_losses
7trainable_variables
8	variables
9	keras_api

:_inbound_nodes

;kernel
<bias
=_outbound_nodes
>regularization_losses
?trainable_variables
@	variables
A	keras_api

B_inbound_nodes

Ckernel
Dbias
E_outbound_nodes
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
|
J_inbound_nodes

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
°

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rate
Uiterm m¡m¢;m£<m¤Cm¥Dm¦Km§Lm¨Vm©WmªXm«v¬v­v®;v¯<v°Cv±Dv²Kv³Lv´VvµWv¶Xv·
 
 
V
0
1
2
V3
W4
X5
;6
<7
C8
D9
K10
L11
V
0
1
2
V3
W4
X5
;6
<7
C8
D9
K10
L11
­

Ylayers
Zmetrics
[non_trainable_variables
\layer_regularization_losses
]layer_metrics
regularization_losses
trainable_variables
	variables
 
 
 
 
 
 
­

^layers
_layer_regularization_losses
`metrics
anon_trainable_variables
blayer_metrics
regularization_losses
trainable_variables
	variables
 
ki
VARIABLE_VALUEmel_stft/real_kernels@layer_with_weights-0/dft_real_kernels/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEmel_stft/imag_kernels@layer_with_weights-0/dft_imag_kernels/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEmel_stft/Variable8layer_with_weights-0/freq2mel/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2

0
1
2
­

clayers
dlayer_regularization_losses
emetrics
fnon_trainable_variables
glayer_metrics
regularization_losses
trainable_variables
	variables
 
 
 
 
 
­

hlayers
ilayer_regularization_losses
jmetrics
knon_trainable_variables
llayer_metrics
"regularization_losses
#trainable_variables
$	variables
 
 
 
 
 
­

mlayers
nlayer_regularization_losses
ometrics
pnon_trainable_variables
qlayer_metrics
(regularization_losses
)trainable_variables
*	variables
~

Vkernel
Wrecurrent_kernel
Xbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
 
 
 
 

V0
W1
X2

V0
W1
X2
¹

vlayers
wlayer_regularization_losses
xnon_trainable_variables
ymetrics
zlayer_metrics
0regularization_losses
1trainable_variables

{states
2	variables
 
 
 
 
 
®

|layers
}layer_regularization_losses
~metrics
non_trainable_variables
layer_metrics
6regularization_losses
7trainable_variables
8	variables
 
[Y
VARIABLE_VALUEdense_37/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_37/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

;0
<1

;0
<1
²
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
>regularization_losses
?trainable_variables
@	variables
 
[Y
VARIABLE_VALUEdense_38/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_38/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

C0
D1

C0
D1
²
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
Fregularization_losses
Gtrainable_variables
H	variables
 
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

K0
L1
²
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
Mregularization_losses
Ntrainable_variables
O	variables
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'simple_rnn_13/simple_rnn_cell_13/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1simple_rnn_13/simple_rnn_cell_13/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%simple_rnn_13/simple_rnn_cell_13/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

V0
W1
X2

V0
W1
X2
²
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
rregularization_losses
strainable_variables
t	variables

,0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables

VARIABLE_VALUEAdam/mel_stft/real_kernels/m\layer_with_weights-0/dft_real_kernels/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/mel_stft/imag_kernels/m\layer_with_weights-0/dft_imag_kernels/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/mel_stft/Variable/mTlayer_with_weights-0/freq2mel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_38/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_38/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_13/simple_rnn_cell_13/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/mel_stft/real_kernels/v\layer_with_weights-0/dft_real_kernels/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/mel_stft/imag_kernels/v\layer_with_weights-0/dft_imag_kernels/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/mel_stft/Variable/vTlayer_with_weights-0/freq2mel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_38/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_38/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_13/simple_rnn_cell_13/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

 serving_default_reshape_13_inputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ}
ú
StatefulPartitionedCallStatefulPartitionedCall serving_default_reshape_13_inputmel_stft/real_kernelsmel_stft/imag_kernelsmel_stft/Variable'simple_rnn_13/simple_rnn_cell_13/kernel%simple_rnn_13/simple_rnn_cell_13/bias1simple_rnn_13/simple_rnn_cell_13/recurrent_kerneldense_37/kerneldense_37/biasdense_38/kerneldense_38/biasdense_39/kerneldense_39/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_480528
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
´
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)mel_stft/real_kernels/Read/ReadVariableOp)mel_stft/imag_kernels/Read/ReadVariableOp%mel_stft/Variable/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp;simple_rnn_13/simple_rnn_cell_13/kernel/Read/ReadVariableOpEsimple_rnn_13/simple_rnn_cell_13/recurrent_kernel/Read/ReadVariableOp9simple_rnn_13/simple_rnn_cell_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0Adam/mel_stft/real_kernels/m/Read/ReadVariableOp0Adam/mel_stft/imag_kernels/m/Read/ReadVariableOp,Adam/mel_stft/Variable/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp*Adam/dense_38/kernel/m/Read/ReadVariableOp(Adam/dense_38/bias/m/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOpBAdam/simple_rnn_13/simple_rnn_cell_13/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_13/simple_rnn_cell_13/bias/m/Read/ReadVariableOp0Adam/mel_stft/real_kernels/v/Read/ReadVariableOp0Adam/mel_stft/imag_kernels/v/Read/ReadVariableOp,Adam/mel_stft/Variable/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOp*Adam/dense_38/kernel/v/Read/ReadVariableOp(Adam/dense_38/bias/v/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOpBAdam/simple_rnn_13/simple_rnn_cell_13/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_13/simple_rnn_cell_13/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_482469
«
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemel_stft/real_kernelsmel_stft/imag_kernelsmel_stft/Variabledense_37/kerneldense_37/biasdense_38/kerneldense_38/biasdense_39/kerneldense_39/biasbeta_1beta_2decaylearning_rate	Adam/iter'simple_rnn_13/simple_rnn_cell_13/kernel1simple_rnn_13/simple_rnn_cell_13/recurrent_kernel%simple_rnn_13/simple_rnn_cell_13/biastotalcounttotal_1count_1Adam/mel_stft/real_kernels/mAdam/mel_stft/imag_kernels/mAdam/mel_stft/Variable/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/dense_38/kernel/mAdam/dense_38/bias/mAdam/dense_39/kernel/mAdam/dense_39/bias/m.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m,Adam/simple_rnn_13/simple_rnn_cell_13/bias/mAdam/mel_stft/real_kernels/vAdam/mel_stft/imag_kernels/vAdam/mel_stft/Variable/vAdam/dense_37/kernel/vAdam/dense_37/bias/vAdam/dense_38/kernel/vAdam/dense_38/bias/vAdam/dense_39/kernel/vAdam/dense_39/bias/v.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_482614¡¶
ç*
ï
while_body_482090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_13_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_13_matmul_readvariableop_resource<
8while_simple_rnn_cell_13_biasadd_readvariableop_resource=
9while_simple_rnn_cell_13_matmul_1_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
.while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype020
.while/simple_rnn_cell_13/MatMul/ReadVariableOpè
while/simple_rnn_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/simple_rnn_cell_13/MatMulÙ
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpå
 while/simple_rnn_cell_13/BiasAddBiasAdd)while/simple_rnn_cell_13/MatMul:product:07while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 while/simple_rnn_cell_13/BiasAddà
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype022
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpÑ
!while/simple_rnn_cell_13/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!while/simple_rnn_cell_13/MatMul_1Ï
while/simple_rnn_cell_13/addAddV2)while/simple_rnn_cell_13/BiasAdd:output:0+while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/add
while/simple_rnn_cell_13/TanhTanh while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/Tanhå
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity!while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_13_biasadd_readvariableop_resource:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_13_matmul_1_readvariableop_resource;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_13_matmul_readvariableop_resource9while_simple_rnn_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Í
ø
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_479343

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ}:ÿÿÿÿÿÿÿÿÿ@::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
µE
ç
-sequential_19_simple_rnn_13_while_body_479205T
Psequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_loop_counterZ
Vsequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_maximum_iterations1
-sequential_19_simple_rnn_13_while_placeholder3
/sequential_19_simple_rnn_13_while_placeholder_13
/sequential_19_simple_rnn_13_while_placeholder_2S
Osequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_strided_slice_1_0
sequential_19_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0Y
Usequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0Z
Vsequential_19_simple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0[
Wsequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0.
*sequential_19_simple_rnn_13_while_identity0
,sequential_19_simple_rnn_13_while_identity_10
,sequential_19_simple_rnn_13_while_identity_20
,sequential_19_simple_rnn_13_while_identity_30
,sequential_19_simple_rnn_13_while_identity_4Q
Msequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_strided_slice_1
sequential_19_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorW
Ssequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceX
Tsequential_19_simple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceY
Usequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceû
Ssequential_19/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2U
Ssequential_19/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeü
Esequential_19/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_19_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0-sequential_19_simple_rnn_13_while_placeholder\sequential_19/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02G
Esequential_19/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem®
Jsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOpUsequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype02L
Jsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOpØ
;sequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMulMatMulLsequential_19/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2=
;sequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul­
Ksequential_19/simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOpVsequential_19_simple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02M
Ksequential_19/simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpÕ
<sequential_19/simple_rnn_13/while/simple_rnn_cell_13/BiasAddBiasAddEsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul:product:0Ssequential_19/simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2>
<sequential_19/simple_rnn_13/while/simple_rnn_cell_13/BiasAdd´
Lsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpWsequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02N
Lsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOpÁ
=sequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1MatMul/sequential_19_simple_rnn_13_while_placeholder_2Tsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2?
=sequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1¿
8sequential_19/simple_rnn_13/while/simple_rnn_cell_13/addAddV2Esequential_19/simple_rnn_13/while/simple_rnn_cell_13/BiasAdd:output:0Gsequential_19/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2:
8sequential_19/simple_rnn_13/while/simple_rnn_cell_13/addî
9sequential_19/simple_rnn_13/while/simple_rnn_cell_13/TanhTanh<sequential_19/simple_rnn_13/while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2;
9sequential_19/simple_rnn_13/while/simple_rnn_cell_13/Tanhñ
Fsequential_19/simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_19_simple_rnn_13_while_placeholder_1-sequential_19_simple_rnn_13_while_placeholder=sequential_19/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02H
Fsequential_19/simple_rnn_13/while/TensorArrayV2Write/TensorListSetItem
'sequential_19/simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/simple_rnn_13/while/add/yÙ
%sequential_19/simple_rnn_13/while/addAddV2-sequential_19_simple_rnn_13_while_placeholder0sequential_19/simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: 2'
%sequential_19/simple_rnn_13/while/add
)sequential_19/simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_19/simple_rnn_13/while/add_1/y
'sequential_19/simple_rnn_13/while/add_1AddV2Psequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_loop_counter2sequential_19/simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_19/simple_rnn_13/while/add_1²
*sequential_19/simple_rnn_13/while/IdentityIdentity+sequential_19/simple_rnn_13/while/add_1:z:0*
T0*
_output_shapes
: 2,
*sequential_19/simple_rnn_13/while/Identityá
,sequential_19/simple_rnn_13/while/Identity_1IdentityVsequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_maximum_iterations*
T0*
_output_shapes
: 2.
,sequential_19/simple_rnn_13/while/Identity_1´
,sequential_19/simple_rnn_13/while/Identity_2Identity)sequential_19/simple_rnn_13/while/add:z:0*
T0*
_output_shapes
: 2.
,sequential_19/simple_rnn_13/while/Identity_2á
,sequential_19/simple_rnn_13/while/Identity_3IdentityVsequential_19/simple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2.
,sequential_19/simple_rnn_13/while/Identity_3Ù
,sequential_19/simple_rnn_13/while/Identity_4Identity=sequential_19/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,sequential_19/simple_rnn_13/while/Identity_4"a
*sequential_19_simple_rnn_13_while_identity3sequential_19/simple_rnn_13/while/Identity:output:0"e
,sequential_19_simple_rnn_13_while_identity_15sequential_19/simple_rnn_13/while/Identity_1:output:0"e
,sequential_19_simple_rnn_13_while_identity_25sequential_19/simple_rnn_13/while/Identity_2:output:0"e
,sequential_19_simple_rnn_13_while_identity_35sequential_19/simple_rnn_13/while/Identity_3:output:0"e
,sequential_19_simple_rnn_13_while_identity_45sequential_19/simple_rnn_13/while/Identity_4:output:0" 
Msequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_strided_slice_1Osequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_strided_slice_1_0"®
Tsequential_19_simple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceVsequential_19_simple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"°
Usequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceWsequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"¬
Ssequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceUsequential_19_simple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0"
sequential_19_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorsequential_19_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ØÙ
ÿ
I__inference_sequential_19_layer_call_and_return_conditional_losses_481486
reshape_13_input0
,mel_stft_convolution_readvariableop_resource2
.mel_stft_convolution_1_readvariableop_resource,
(mel_stft_shape_1_readvariableop_resourceC
?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resourceD
@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceE
Asimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource
identity¢simple_rnn_13/whiled
reshape_13/ShapeShapereshape_13_input*
T0*
_output_shapes
:2
reshape_13/Shape
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2¤
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_13/Reshape/shape/2×
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape
reshape_13/ReshapeReshapereshape_13_input!reshape_13/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
reshape_13/Reshape
mel_stft/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
mel_stft/strided_slice/stack
mel_stft/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
mel_stft/strided_slice/stack_1
mel_stft/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/strided_slice/stack_2¼
mel_stft/strided_sliceStridedSlicereshape_13/Reshape:output:0%mel_stft/strided_slice/stack:output:0'mel_stft/strided_slice/stack_1:output:0'mel_stft/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask2
mel_stft/strided_slice
mel_stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mel_stft/transpose/perm¯
mel_stft/transpose	Transposemel_stft/strided_slice:output:0 mel_stft/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transposet
mel_stft/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/ExpandDims/dim­
mel_stft/ExpandDims
ExpandDimsmel_stft/transpose:y:0 mel_stft/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/ExpandDimsÁ
#mel_stft/convolution/ReadVariableOpReadVariableOp,mel_stft_convolution_readvariableop_resource*(
_output_shapes
:*
dtype02%
#mel_stft/convolution/ReadVariableOpå
mel_stft/convolutionConv2Dmel_stft/ExpandDims:output:0+mel_stft/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolutionÇ
%mel_stft/convolution_1/ReadVariableOpReadVariableOp.mel_stft_convolution_1_readvariableop_resource*(
_output_shapes
:*
dtype02'
%mel_stft/convolution_1/ReadVariableOpë
mel_stft/convolution_1Conv2Dmel_stft/ExpandDims:output:0-mel_stft/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolution_1e
mel_stft/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow/y
mel_stft/powPowmel_stft/convolution:output:0mel_stft/pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/powi
mel_stft/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow_1/y
mel_stft/pow_1Powmel_stft/convolution_1:output:0mel_stft/pow_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/pow_1
mel_stft/addAddV2mel_stft/pow:z:0mel_stft/pow_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/add
mel_stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_1/permª
mel_stft/transpose_1	Transposemel_stft/add:z:0"mel_stft/transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_1
mel_stft/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_2/perm²
mel_stft/transpose_2	Transposemel_stft/transpose_1:y:0"mel_stft/transpose_2/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_2h
mel_stft/ShapeShapemel_stft/transpose_2:y:0*
T0*
_output_shapes
:2
mel_stft/Shapey
mel_stft/unstackUnpackmel_stft/Shape:output:0*
T0*
_output_shapes

: : : : *	
num2
mel_stft/unstack¬
mel_stft/Shape_1/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02!
mel_stft/Shape_1/ReadVariableOpu
mel_stft/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"  P   2
mel_stft/Shape_1{
mel_stft/unstack_1Unpackmel_stft/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
mel_stft/unstack_1
mel_stft/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
mel_stft/Reshape/shape
mel_stft/ReshapeReshapemel_stft/transpose_2:y:0mel_stft/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mel_stft/Reshape´
#mel_stft/transpose_3/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02%
#mel_stft/transpose_3/ReadVariableOp
mel_stft/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
mel_stft/transpose_3/perm´
mel_stft/transpose_3	Transpose+mel_stft/transpose_3/ReadVariableOp:value:0"mel_stft/transpose_3/perm:output:0*
T0*
_output_shapes
:	P2
mel_stft/transpose_3
mel_stft/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ÿÿÿÿ2
mel_stft/Reshape_1/shape
mel_stft/Reshape_1Reshapemel_stft/transpose_3:y:0!mel_stft/Reshape_1/shape:output:0*
T0*
_output_shapes
:	P2
mel_stft/Reshape_1
mel_stft/MatMulMatMulmel_stft/Reshape:output:0mel_stft/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP2
mel_stft/MatMulz
mel_stft/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/Reshape_2/shape/1z
mel_stft/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}2
mel_stft/Reshape_2/shape/2z
mel_stft/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :P2
mel_stft/Reshape_2/shape/3ô
mel_stft/Reshape_2/shapePackmel_stft/unstack:output:0#mel_stft/Reshape_2/shape/1:output:0#mel_stft/Reshape_2/shape/2:output:0#mel_stft/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2
mel_stft/Reshape_2/shape«
mel_stft/Reshape_2Reshapemel_stft/MatMul:product:0!mel_stft/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P2
mel_stft/Reshape_2
mel_stft/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_4/perm´
mel_stft/transpose_4	Transposemel_stft/Reshape_2:output:0"mel_stft/transpose_4/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/transpose_4e
mel_stft/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mel_stft/Consti
mel_stft/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
mel_stft/Const_1º
mel_stft/clip_by_value/MinimumMinimummel_stft/transpose_4:y:0mel_stft/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
mel_stft/clip_by_value/Minimum²
mel_stft/clip_by_valueMaximum"mel_stft/clip_by_value/Minimum:z:0mel_stft/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/clip_by_value|
mel_stft/SqrtSqrtmel_stft/clip_by_value:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Sqrti
mel_stft/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
mel_stft/Pow_2/y
mel_stft/Pow_2Powmel_stft/Sqrt:y:0mel_stft/Pow_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Pow_2m
mel_stft/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
mel_stft/Maximum/y
mel_stft/MaximumMaximummel_stft/Pow_2:z:0mel_stft/Maximum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximums
mel_stft/LogLogmel_stft/Maximum:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Loge
mel_stft/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mel_stft/mul/x
mel_stft/mulMulmel_stft/mul/x:output:0mel_stft/Log:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/mulm
mel_stft/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *]@2
mel_stft/truediv/y
mel_stft/truedivRealDivmel_stft/mul:z:0mel_stft/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/truediv
mel_stft/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/Max/reduction_indices­
mel_stft/MaxMaxmel_stft/truediv:z:0'mel_stft/Max/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
mel_stft/Max
mel_stft/subSubmel_stft/truediv:z:0mel_stft/Max:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/subq
mel_stft/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â2
mel_stft/Maximum_1/y
mel_stft/Maximum_1Maximummel_stft/sub:z:0mel_stft/Maximum_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximum_1©
(normalization2d_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2*
(normalization2d_7/Mean/reduction_indicesÎ
normalization2d_7/MeanMeanmel_stft/Maximum_1:z:01normalization2d_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
normalization2d_7/Meanß
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2E
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices
1normalization2d_7/reduce_std/reduce_variance/MeanMeanmel_stft/Maximum_1:z:0Lnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(23
1normalization2d_7/reduce_std/reduce_variance/Meanù
0normalization2d_7/reduce_std/reduce_variance/subSubmel_stft/Maximum_1:z:0:normalization2d_7/reduce_std/reduce_variance/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}22
0normalization2d_7/reduce_std/reduce_variance/subä
3normalization2d_7/reduce_std/reduce_variance/SquareSquare4normalization2d_7/reduce_std/reduce_variance/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}25
3normalization2d_7/reduce_std/reduce_variance/Squareã
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2G
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesÆ
3normalization2d_7/reduce_std/reduce_variance/Mean_1Mean7normalization2d_7/reduce_std/reduce_variance/Square:y:0Nnormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(25
3normalization2d_7/reduce_std/reduce_variance/Mean_1Æ
!normalization2d_7/reduce_std/SqrtSqrt<normalization2d_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!normalization2d_7/reduce_std/Sqrt¨
normalization2d_7/subSubmel_stft/Maximum_1:z:0normalization2d_7/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/subw
normalization2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
normalization2d_7/add/yº
normalization2d_7/addAddV2%normalization2d_7/reduce_std/Sqrt:y:0 normalization2d_7/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization2d_7/add±
normalization2d_7/truedivRealDivnormalization2d_7/sub:z:0normalization2d_7/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/truediv´
squeeze_last_dim/SqueezeSqueezenormalization2d_7/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
squeeze_last_dim/Squeeze{
simple_rnn_13/ShapeShape!squeeze_last_dim/Squeeze:output:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape
!simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_13/strided_slice/stack
#simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_1
#simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_2¶
simple_rnn_13/strided_sliceStridedSlicesimple_rnn_13/Shape:output:0*simple_rnn_13/strided_slice/stack:output:0,simple_rnn_13/strided_slice/stack_1:output:0,simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slicex
simple_rnn_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/mul/y¤
simple_rnn_13/zeros/mulMul$simple_rnn_13/strided_slice:output:0"simple_rnn_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/mul{
simple_rnn_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
simple_rnn_13/zeros/Less/y
simple_rnn_13/zeros/LessLesssimple_rnn_13/zeros/mul:z:0#simple_rnn_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/Less~
simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/packed/1»
simple_rnn_13/zeros/packedPack$simple_rnn_13/strided_slice:output:0%simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_13/zeros/packed{
simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_13/zeros/Const­
simple_rnn_13/zerosFill#simple_rnn_13/zeros/packed:output:0"simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_13/zeros
simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_13/transpose/perm¿
simple_rnn_13/transpose	Transpose!squeeze_last_dim/Squeeze:output:0%simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
simple_rnn_13/transposey
simple_rnn_13/Shape_1Shapesimple_rnn_13/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape_1
#simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_1/stack
%simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_1
%simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_2Â
simple_rnn_13/strided_slice_1StridedSlicesimple_rnn_13/Shape_1:output:0,simple_rnn_13/strided_slice_1/stack:output:0.simple_rnn_13/strided_slice_1/stack_1:output:0.simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slice_1¡
)simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)simple_rnn_13/TensorArrayV2/element_shapeê
simple_rnn_13/TensorArrayV2TensorListReserve2simple_rnn_13/TensorArrayV2/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2Û
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2E
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape°
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_13/transpose:y:0Lsimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensor
#simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_2/stack
%simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_1
%simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_2Ð
simple_rnn_13/strided_slice_2StridedSlicesimple_rnn_13/transpose:y:0,simple_rnn_13/strided_slice_2/stack:output:0.simple_rnn_13/strided_slice_2/stack_1:output:0.simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
simple_rnn_13/strided_slice_2ð
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype028
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpö
'simple_rnn_13/simple_rnn_cell_13/MatMulMatMul&simple_rnn_13/strided_slice_2:output:0>simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'simple_rnn_13/simple_rnn_cell_13/MatMulï
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp
(simple_rnn_13/simple_rnn_cell_13/BiasAddBiasAdd1simple_rnn_13/simple_rnn_cell_13/MatMul:product:0?simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(simple_rnn_13/simple_rnn_cell_13/BiasAddö
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpò
)simple_rnn_13/simple_rnn_cell_13/MatMul_1MatMulsimple_rnn_13/zeros:output:0@simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)simple_rnn_13/simple_rnn_cell_13/MatMul_1ï
$simple_rnn_13/simple_rnn_cell_13/addAddV21simple_rnn_13/simple_rnn_cell_13/BiasAdd:output:03simple_rnn_13/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$simple_rnn_13/simple_rnn_cell_13/add²
%simple_rnn_13/simple_rnn_cell_13/TanhTanh(simple_rnn_13/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%simple_rnn_13/simple_rnn_cell_13/Tanh«
+simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2-
+simple_rnn_13/TensorArrayV2_1/element_shapeð
simple_rnn_13/TensorArrayV2_1TensorListReserve4simple_rnn_13/TensorArrayV2_1/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2_1j
simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_13/time
&simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&simple_rnn_13/while/maximum_iterations
 simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_13/while/loop_counter
simple_rnn_13/whileWhile)simple_rnn_13/while/loop_counter:output:0/simple_rnn_13/while/maximum_iterations:output:0simple_rnn_13/time:output:0&simple_rnn_13/TensorArrayV2_1:handle:0simple_rnn_13/zeros:output:0&simple_rnn_13/strided_slice_1:output:0Esimple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_13_while_body_481397*+
cond#R!
simple_rnn_13_while_cond_481396*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
simple_rnn_13/whileÑ
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2@
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape 
0simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_13/while:output:3Gsimple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype022
0simple_rnn_13/TensorArrayV2Stack/TensorListStack
#simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2%
#simple_rnn_13/strided_slice_3/stack
%simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_13/strided_slice_3/stack_1
%simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_3/stack_2î
simple_rnn_13/strided_slice_3StridedSlice9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_13/strided_slice_3/stack:output:0.simple_rnn_13/strided_slice_3/stack_1:output:0.simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
simple_rnn_13/strided_slice_3
simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_13/transpose_1/permÝ
simple_rnn_13/transpose_1	Transpose9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
simple_rnn_13/transpose_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_8/Const
flatten_8/ReshapeReshapesimple_rnn_13/transpose_1:y:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
flatten_8/Reshape©
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype02 
dense_37/MatMul/ReadVariableOp¢
dense_37/MatMulMatMulflatten_8/Reshape:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/MatMul§
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp¥
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/BiasAdds
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/Relu¨
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_38/MatMul/ReadVariableOp£
dense_38/MatMulMatMuldense_37/Relu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/MatMul§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp¥
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/BiasAdds
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Relu¨
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

: $*
dtype02 
dense_39/MatMul/ReadVariableOp£
dense_39/MatMulMatMuldense_38/Relu:activations:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/Softmax
IdentityIdentitydense_39/Softmax:softmax:0^simple_rnn_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::2*
simple_rnn_13/whilesimple_rnn_13/while:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
*
_user_specified_namereshape_13_input
ù
h
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_479957

inputs
identity{
SqueezeSqueezeinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
®	

.__inference_sequential_19_layer_call_fn_481007

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4803962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
à
d
M__inference_normalization2d_7_layer_call_and_return_conditional_losses_481661
x
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices
MeanMeanxMean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Mean»
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indicesÔ
reduce_std/reduce_variance/MeanMeanx:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2!
reduce_std/reduce_variance/Mean®
reduce_std/reduce_variance/subSubx(reduce_std/reduce_variance/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
reduce_std/reduce_variance/sub®
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2#
!reduce_std/reduce_variance/Square¿
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indicesþ
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reduce_std/Sqrt]
subSubxMean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addi
truedivRealDivsub:z:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2	
truedivg
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}

_user_specified_namex
ç*
ï
while_body_481844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_13_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_13_matmul_readvariableop_resource<
8while_simple_rnn_cell_13_biasadd_readvariableop_resource=
9while_simple_rnn_cell_13_matmul_1_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
.while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype020
.while/simple_rnn_cell_13/MatMul/ReadVariableOpè
while/simple_rnn_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/simple_rnn_cell_13/MatMulÙ
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpå
 while/simple_rnn_cell_13/BiasAddBiasAdd)while/simple_rnn_cell_13/MatMul:product:07while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 while/simple_rnn_cell_13/BiasAddà
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype022
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpÑ
!while/simple_rnn_cell_13/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!while/simple_rnn_cell_13/MatMul_1Ï
while/simple_rnn_cell_13/addAddV2)while/simple_rnn_cell_13/BiasAdd:output:0+while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/add
while/simple_rnn_cell_13/TanhTanh while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/Tanhå
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity!while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_13_biasadd_readvariableop_resource:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_13_matmul_1_readvariableop_resource;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_13_matmul_readvariableop_resource9while_simple_rnn_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
>
×
D__inference_mel_stft_layer_call_and_return_conditional_losses_479902
x'
#convolution_readvariableop_resource)
%convolution_1_readvariableop_resource#
shape_1_readvariableop_resource
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2õ
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask2
strided_sliceu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposestrided_slice:output:0transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimstranspose:y:0ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2

ExpandDims¦
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*(
_output_shapes
:*
dtype02
convolution/ReadVariableOpÁ
convolutionConv2DExpandDims:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
convolution¬
convolution_1/ReadVariableOpReadVariableOp%convolution_1_readvariableop_resource*(
_output_shapes
:*
dtype02
convolution_1/ReadVariableOpÇ
convolution_1Conv2DExpandDims:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
convolution_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yr
powPowconvolution:output:0pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
powW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yz
pow_1Powconvolution_1:output:0pow_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
pow_1b
addAddV2pow:z:0	pow_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
add}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/perm
transpose_1	Transposeadd:z:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
transpose_1}
transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_2/perm
transpose_2	Transposetranspose_1:y:0transpose_2/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
transpose_2M
ShapeShapetranspose_2:y:0*
T0*
_output_shapes
:2
Shape^
unstackUnpackShape:output:0*
T0*
_output_shapes

: : : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"  P   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Reshape/shapey
ReshapeReshapetranspose_2:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshape
transpose_3/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes
:	P2
transpose_3s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ÿÿÿÿ2
Reshape_1/shapev
	Reshape_1Reshapetranspose_3:y:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	P2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}2
Reshape_2/shape/2h
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :P2
Reshape_2/shape/3¾
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P2
	Reshape_2}
transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_4/perm
transpose_4	TransposeReshape_2:output:0transpose_4/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
transpose_4S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1
clip_by_value/MinimumMinimumtranspose_4:y:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
clip_by_value/Minimum
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
clip_by_valuea
SqrtSqrtclip_by_value:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
SqrtW
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
Pow_2/yk
Pow_2PowSqrt:y:0Pow_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
Pow_2[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
	Maximum/yv
MaximumMaximum	Pow_2:z:0Maximum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2	
MaximumX
LogLogMaximum:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
LogS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/xd
mulMulmul/x:output:0Log:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *]@2
	truediv/yt
truedivRealDivmul:z:0truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2	
truediv
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Max/reduction_indices
MaxMaxtruediv:z:0Max/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Maxf
subSubtruediv:z:0Max:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
sub_
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â2
Maximum_1/yz
	Maximum_1Maximumsub:z:0Maximum_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
	Maximum_1i
IdentityIdentityMaximum_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}::::O K
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}

_user_specified_namex
ü<
ú
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_479680

inputs
simple_rnn_cell_13_479605
simple_rnn_cell_13_479607
simple_rnn_cell_13_479609
identity¢*simple_rnn_cell_13/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2
*simple_rnn_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_13_479605simple_rnn_cell_13_479607simple_rnn_cell_13_479609*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_4793432,
*simple_rnn_cell_13/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterü
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_13_479605simple_rnn_cell_13_479607simple_rnn_cell_13_479609*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_479617*
condR
while_cond_479616*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1¥
IdentityIdentitytranspose_1:y:0+^simple_rnn_cell_13/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}:::2X
*simple_rnn_cell_13/StatefulPartitionedCall*simple_rnn_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
è

´
simple_rnn_13_while_cond_4806638
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_2:
6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480663___redundant_placeholder0P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480663___redundant_placeholder1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480663___redundant_placeholder2P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480663___redundant_placeholder3 
simple_rnn_13_while_identity
¶
simple_rnn_13/while/LessLesssimple_rnn_13_while_placeholder6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_13/while/Less
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_13/while/Identity"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
8
é	
simple_rnn_13_while_body_4806648
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_27
3simple_rnn_13_while_simple_rnn_13_strided_slice_1_0s
osimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0L
Hsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0M
Isimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0 
simple_rnn_13_while_identity"
simple_rnn_13_while_identity_1"
simple_rnn_13_while_identity_2"
simple_rnn_13_while_identity_3"
simple_rnn_13_while_identity_45
1simple_rnn_13_while_simple_rnn_13_strided_slice_1q
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceJ
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceK
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceß
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2G
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape§
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_13_while_placeholderNsimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype029
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype02>
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp 
-simple_rnn_13/while/simple_rnn_cell_13/MatMulMatMul>simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2/
-simple_rnn_13/while/simple_rnn_cell_13/MatMul
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02?
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp
.simple_rnn_13/while/simple_rnn_cell_13/BiasAddBiasAdd7simple_rnn_13/while/simple_rnn_cell_13/MatMul:product:0Esimple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.simple_rnn_13/while/simple_rnn_cell_13/BiasAdd
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02@
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1MatMul!simple_rnn_13_while_placeholder_2Fsimple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@21
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1
*simple_rnn_13/while/simple_rnn_cell_13/addAddV27simple_rnn_13/while/simple_rnn_cell_13/BiasAdd:output:09simple_rnn_13/while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*simple_rnn_13/while/simple_rnn_cell_13/addÄ
+simple_rnn_13/while/simple_rnn_cell_13/TanhTanh.simple_rnn_13/while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+simple_rnn_13/while/simple_rnn_cell_13/Tanh«
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_13_while_placeholder_1simple_rnn_13_while_placeholder/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add/y¡
simple_rnn_13/while/addAddV2simple_rnn_13_while_placeholder"simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add|
simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add_1/y¼
simple_rnn_13/while/add_1AddV24simple_rnn_13_while_simple_rnn_13_while_loop_counter$simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add_1
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/Identity©
simple_rnn_13/while/Identity_1Identity:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_1
simple_rnn_13/while/Identity_2Identitysimple_rnn_13/while/add:z:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_2·
simple_rnn_13/while/Identity_3IdentityHsimple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_3¯
simple_rnn_13/while/Identity_4Identity/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
simple_rnn_13/while/Identity_4"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0"I
simple_rnn_13_while_identity_1'simple_rnn_13/while/Identity_1:output:0"I
simple_rnn_13_while_identity_2'simple_rnn_13/while/Identity_2:output:0"I
simple_rnn_13_while_identity_3'simple_rnn_13/while/Identity_3:output:0"I
simple_rnn_13_while_identity_4'simple_rnn_13/while/Identity_4:output:0"h
1simple_rnn_13_while_simple_rnn_13_strided_slice_13simple_rnn_13_while_simple_rnn_13_strided_slice_1_0"
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0"à
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
8
é	
simple_rnn_13_while_body_4808898
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_27
3simple_rnn_13_while_simple_rnn_13_strided_slice_1_0s
osimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0L
Hsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0M
Isimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0 
simple_rnn_13_while_identity"
simple_rnn_13_while_identity_1"
simple_rnn_13_while_identity_2"
simple_rnn_13_while_identity_3"
simple_rnn_13_while_identity_45
1simple_rnn_13_while_simple_rnn_13_strided_slice_1q
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceJ
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceK
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceß
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2G
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape§
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_13_while_placeholderNsimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype029
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype02>
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp 
-simple_rnn_13/while/simple_rnn_cell_13/MatMulMatMul>simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2/
-simple_rnn_13/while/simple_rnn_cell_13/MatMul
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02?
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp
.simple_rnn_13/while/simple_rnn_cell_13/BiasAddBiasAdd7simple_rnn_13/while/simple_rnn_cell_13/MatMul:product:0Esimple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.simple_rnn_13/while/simple_rnn_cell_13/BiasAdd
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02@
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1MatMul!simple_rnn_13_while_placeholder_2Fsimple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@21
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1
*simple_rnn_13/while/simple_rnn_cell_13/addAddV27simple_rnn_13/while/simple_rnn_cell_13/BiasAdd:output:09simple_rnn_13/while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*simple_rnn_13/while/simple_rnn_cell_13/addÄ
+simple_rnn_13/while/simple_rnn_cell_13/TanhTanh.simple_rnn_13/while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+simple_rnn_13/while/simple_rnn_cell_13/Tanh«
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_13_while_placeholder_1simple_rnn_13_while_placeholder/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add/y¡
simple_rnn_13/while/addAddV2simple_rnn_13_while_placeholder"simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add|
simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add_1/y¼
simple_rnn_13/while/add_1AddV24simple_rnn_13_while_simple_rnn_13_while_loop_counter$simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add_1
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/Identity©
simple_rnn_13/while/Identity_1Identity:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_1
simple_rnn_13/while/Identity_2Identitysimple_rnn_13/while/add:z:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_2·
simple_rnn_13/while/Identity_3IdentityHsimple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_3¯
simple_rnn_13/while/Identity_4Identity/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
simple_rnn_13/while/Identity_4"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0"I
simple_rnn_13_while_identity_1'simple_rnn_13/while/Identity_1:output:0"I
simple_rnn_13_while_identity_2'simple_rnn_13/while/Identity_2:output:0"I
simple_rnn_13_while_identity_3'simple_rnn_13/while/Identity_3:output:0"I
simple_rnn_13_while_identity_4'simple_rnn_13/while/Identity_4:output:0"h
1simple_rnn_13_while_simple_rnn_13_strided_slice_13simple_rnn_13_while_simple_rnn_13_strided_slice_1_0"
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0"à
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ùC

I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_482156

inputs5
1simple_rnn_cell_13_matmul_readvariableop_resource6
2simple_rnn_cell_13_biasadd_readvariableop_resource7
3simple_rnn_cell_13_matmul_1_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2Æ
(simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype02*
(simple_rnn_cell_13/MatMul/ReadVariableOp¾
simple_rnn_cell_13/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMulÅ
)simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)simple_rnn_cell_13/BiasAdd/ReadVariableOpÍ
simple_rnn_cell_13/BiasAddBiasAdd#simple_rnn_cell_13/MatMul:product:01simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/BiasAddÌ
*simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*simple_rnn_cell_13/MatMul_1/ReadVariableOpº
simple_rnn_cell_13/MatMul_1MatMulzeros:output:02simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMul_1·
simple_rnn_cell_13/addAddV2#simple_rnn_cell_13/BiasAdd:output:0%simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/add
simple_rnn_cell_13/TanhTanhsimple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_13_matmul_readvariableop_resource2simple_rnn_cell_13_biasadd_readvariableop_resource3simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_482090*
condR
while_cond_482089*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
transpose_1o
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿP}:::2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
þ
¾
-sequential_19_simple_rnn_13_while_cond_479204T
Psequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_loop_counterZ
Vsequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_maximum_iterations1
-sequential_19_simple_rnn_13_while_placeholder3
/sequential_19_simple_rnn_13_while_placeholder_13
/sequential_19_simple_rnn_13_while_placeholder_2V
Rsequential_19_simple_rnn_13_while_less_sequential_19_simple_rnn_13_strided_slice_1l
hsequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_cond_479204___redundant_placeholder0l
hsequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_cond_479204___redundant_placeholder1l
hsequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_cond_479204___redundant_placeholder2l
hsequential_19_simple_rnn_13_while_sequential_19_simple_rnn_13_while_cond_479204___redundant_placeholder3.
*sequential_19_simple_rnn_13_while_identity
ü
&sequential_19/simple_rnn_13/while/LessLess-sequential_19_simple_rnn_13_while_placeholderRsequential_19_simple_rnn_13_while_less_sequential_19_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: 2(
&sequential_19/simple_rnn_13/while/Less±
*sequential_19/simple_rnn_13/while/IdentityIdentity*sequential_19/simple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: 2,
*sequential_19/simple_rnn_13/while/Identity"a
*sequential_19_simple_rnn_13_while_identity3sequential_19/simple_rnn_13/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ì
b
F__inference_reshape_13_layer_call_and_return_conditional_losses_481557

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1m
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
½
M
1__inference_squeeze_last_dim_layer_call_fn_481686

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_4799572
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
©
¬
D__inference_dense_38_layer_call_and_return_conditional_losses_482220

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï	
µ
3__inference_simple_rnn_cell_13_layer_call_fn_482297

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_4793432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ}:ÿÿÿÿÿÿÿÿÿ@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
ùC

I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_480081

inputs5
1simple_rnn_cell_13_matmul_readvariableop_resource6
2simple_rnn_cell_13_biasadd_readvariableop_resource7
3simple_rnn_cell_13_matmul_1_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2Æ
(simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype02*
(simple_rnn_cell_13/MatMul/ReadVariableOp¾
simple_rnn_cell_13/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMulÅ
)simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)simple_rnn_cell_13/BiasAdd/ReadVariableOpÍ
simple_rnn_cell_13/BiasAddBiasAdd#simple_rnn_cell_13/MatMul:product:01simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/BiasAddÌ
*simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*simple_rnn_cell_13/MatMul_1/ReadVariableOpº
simple_rnn_cell_13/MatMul_1MatMulzeros:output:02simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMul_1·
simple_rnn_cell_13/addAddV2#simple_rnn_cell_13/BiasAdd:output:0%simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/add
simple_rnn_cell_13/TanhTanhsimple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_13_matmul_readvariableop_resource2simple_rnn_cell_13_biasadd_readvariableop_resource3simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_480015*
condR
while_cond_480014*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
transpose_1o
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿP}:::2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
Ð
ª
while_cond_481843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_481843___redundant_placeholder04
0while_while_cond_481843___redundant_placeholder14
0while_while_cond_481843___redundant_placeholder24
0while_while_cond_481843___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
½
M
1__inference_squeeze_last_dim_layer_call_fn_481681

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_4799522
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
à
d
M__inference_normalization2d_7_layer_call_and_return_conditional_losses_479939
x
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices
MeanMeanxMean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Mean»
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indicesÔ
reduce_std/reduce_variance/MeanMeanx:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2!
reduce_std/reduce_variance/Mean®
reduce_std/reduce_variance/subSubx(reduce_std/reduce_variance/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
reduce_std/reduce_variance/sub®
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2#
!reduce_std/reduce_variance/Square¿
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indicesþ
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reduce_std/Sqrt]
subSubxMean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addi
truedivRealDivsub:z:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2	
truedivg
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}

_user_specified_namex
ç*
ï
while_body_480015
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_13_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_13_matmul_readvariableop_resource<
8while_simple_rnn_cell_13_biasadd_readvariableop_resource=
9while_simple_rnn_cell_13_matmul_1_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
.while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype020
.while/simple_rnn_cell_13/MatMul/ReadVariableOpè
while/simple_rnn_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/simple_rnn_cell_13/MatMulÙ
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpå
 while/simple_rnn_cell_13/BiasAddBiasAdd)while/simple_rnn_cell_13/MatMul:product:07while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 while/simple_rnn_cell_13/BiasAddà
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype022
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpÑ
!while/simple_rnn_cell_13/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!while/simple_rnn_cell_13/MatMul_1Ï
while/simple_rnn_cell_13/addAddV2)while/simple_rnn_cell_13/BiasAdd:output:0+while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/add
while/simple_rnn_cell_13/TanhTanh while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/Tanhå
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity!while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_13_biasadd_readvariableop_resource:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_13_matmul_1_readvariableop_resource;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_13_matmul_readvariableop_resource9while_simple_rnn_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ð
ª
while_cond_480014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_480014___redundant_placeholder04
0while_while_cond_480014___redundant_placeholder14
0while_while_cond_480014___redundant_placeholder24
0while_while_cond_480014___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ð
ª
while_cond_479733
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_479733___redundant_placeholder04
0while_while_cond_479733___redundant_placeholder14
0while_while_cond_479733___redundant_placeholder24
0while_while_cond_479733___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ì	
¤
.__inference_sequential_19_layer_call_fn_481515
reshape_13_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallreshape_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4803962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
*
_user_specified_namereshape_13_input


)__inference_mel_stft_layer_call_fn_481644
x
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mel_stft_layer_call_and_return_conditional_losses_4799022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}

_user_specified_namex
©
¬
D__inference_dense_38_layer_call_and_return_conditional_losses_480275

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
±
¬
D__inference_dense_39_layer_call_and_return_conditional_losses_480302

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: $*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
ú
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_482266

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ}:ÿÿÿÿÿÿÿÿÿ@::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
á
~
)__inference_dense_37_layer_call_fn_482209

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4802482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¼b
Ä
__inference__traced_save_482469
file_prefix4
0savev2_mel_stft_real_kernels_read_readvariableop4
0savev2_mel_stft_imag_kernels_read_readvariableop0
,savev2_mel_stft_variable_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	F
Bsavev2_simple_rnn_13_simple_rnn_cell_13_kernel_read_readvariableopP
Lsavev2_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_13_simple_rnn_cell_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_adam_mel_stft_real_kernels_m_read_readvariableop;
7savev2_adam_mel_stft_imag_kernels_m_read_readvariableop7
3savev2_adam_mel_stft_variable_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop5
1savev2_adam_dense_38_kernel_m_read_readvariableop3
/savev2_adam_dense_38_bias_m_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_m_read_readvariableop;
7savev2_adam_mel_stft_real_kernels_v_read_readvariableop;
7savev2_adam_mel_stft_imag_kernels_v_read_readvariableop7
3savev2_adam_mel_stft_variable_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop5
1savev2_adam_dense_38_kernel_v_read_readvariableop3
/savev2_adam_dense_38_bias_v_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a6c679294a2549198b0ebc5e2fbdfaff/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameØ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*ê
valueàBÝ.B@layer_with_weights-0/dft_real_kernels/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/dft_imag_kernels/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/freq2mel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_real_kernels/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_imag_kernels/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/freq2mel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_real_kernels/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_imag_kernels/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/freq2mel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesä
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_mel_stft_real_kernels_read_readvariableop0savev2_mel_stft_imag_kernels_read_readvariableop,savev2_mel_stft_variable_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableopBsavev2_simple_rnn_13_simple_rnn_cell_13_kernel_read_readvariableopLsavev2_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_read_readvariableop@savev2_simple_rnn_13_simple_rnn_cell_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_adam_mel_stft_real_kernels_m_read_readvariableop7savev2_adam_mel_stft_imag_kernels_m_read_readvariableop3savev2_adam_mel_stft_variable_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop1savev2_adam_dense_38_kernel_m_read_readvariableop/savev2_adam_dense_38_bias_m_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableopIsavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_m_read_readvariableop7savev2_adam_mel_stft_real_kernels_v_read_readvariableop7savev2_adam_mel_stft_imag_kernels_v_read_readvariableop3savev2_adam_mel_stft_variable_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableop1savev2_adam_dense_38_kernel_v_read_readvariableop/savev2_adam_dense_38_bias_v_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableopIsavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*¥
_input_shapes
: :::	P:	(@:@:@ : : $:$: : : : : :}@:@@:@: : : : :::	P:	(@:@:@ : : $:$:}@:@@:@:::	P:	(@:@:@ : : $:$:}@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:.*
(
_output_shapes
::.*
(
_output_shapes
::%!

_output_shapes
:	P:%!

_output_shapes
:	(@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: $: 	

_output_shapes
:$:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:}@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
::.*
(
_output_shapes
::%!

_output_shapes
:	P:%!

_output_shapes
:	(@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: $: 

_output_shapes
:$:$ 

_output_shapes

:}@:$  

_output_shapes

:@@: !

_output_shapes
:@:."*
(
_output_shapes
::.#*
(
_output_shapes
::%$!

_output_shapes
:	P:%%!

_output_shapes
:	(@: &

_output_shapes
:@:$' 

_output_shapes

:@ : (

_output_shapes
: :$) 

_output_shapes

: $: *

_output_shapes
:$:$+ 

_output_shapes

:}@:$, 

_output_shapes

:@@: -

_output_shapes
:@:.

_output_shapes
: 
ù
h
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_481671

inputs
identity{
SqueezeSqueezeinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
°Ù
õ
I__inference_sequential_19_layer_call_and_return_conditional_losses_480753

inputs0
,mel_stft_convolution_readvariableop_resource2
.mel_stft_convolution_1_readvariableop_resource,
(mel_stft_shape_1_readvariableop_resourceC
?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resourceD
@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceE
Asimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource
identity¢simple_rnn_13/whileZ
reshape_13/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_13/Shape
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2¤
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_13/Reshape/shape/2×
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape
reshape_13/ReshapeReshapeinputs!reshape_13/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
reshape_13/Reshape
mel_stft/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
mel_stft/strided_slice/stack
mel_stft/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
mel_stft/strided_slice/stack_1
mel_stft/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/strided_slice/stack_2¼
mel_stft/strided_sliceStridedSlicereshape_13/Reshape:output:0%mel_stft/strided_slice/stack:output:0'mel_stft/strided_slice/stack_1:output:0'mel_stft/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask2
mel_stft/strided_slice
mel_stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mel_stft/transpose/perm¯
mel_stft/transpose	Transposemel_stft/strided_slice:output:0 mel_stft/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transposet
mel_stft/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/ExpandDims/dim­
mel_stft/ExpandDims
ExpandDimsmel_stft/transpose:y:0 mel_stft/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/ExpandDimsÁ
#mel_stft/convolution/ReadVariableOpReadVariableOp,mel_stft_convolution_readvariableop_resource*(
_output_shapes
:*
dtype02%
#mel_stft/convolution/ReadVariableOpå
mel_stft/convolutionConv2Dmel_stft/ExpandDims:output:0+mel_stft/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolutionÇ
%mel_stft/convolution_1/ReadVariableOpReadVariableOp.mel_stft_convolution_1_readvariableop_resource*(
_output_shapes
:*
dtype02'
%mel_stft/convolution_1/ReadVariableOpë
mel_stft/convolution_1Conv2Dmel_stft/ExpandDims:output:0-mel_stft/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolution_1e
mel_stft/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow/y
mel_stft/powPowmel_stft/convolution:output:0mel_stft/pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/powi
mel_stft/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow_1/y
mel_stft/pow_1Powmel_stft/convolution_1:output:0mel_stft/pow_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/pow_1
mel_stft/addAddV2mel_stft/pow:z:0mel_stft/pow_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/add
mel_stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_1/permª
mel_stft/transpose_1	Transposemel_stft/add:z:0"mel_stft/transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_1
mel_stft/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_2/perm²
mel_stft/transpose_2	Transposemel_stft/transpose_1:y:0"mel_stft/transpose_2/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_2h
mel_stft/ShapeShapemel_stft/transpose_2:y:0*
T0*
_output_shapes
:2
mel_stft/Shapey
mel_stft/unstackUnpackmel_stft/Shape:output:0*
T0*
_output_shapes

: : : : *	
num2
mel_stft/unstack¬
mel_stft/Shape_1/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02!
mel_stft/Shape_1/ReadVariableOpu
mel_stft/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"  P   2
mel_stft/Shape_1{
mel_stft/unstack_1Unpackmel_stft/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
mel_stft/unstack_1
mel_stft/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
mel_stft/Reshape/shape
mel_stft/ReshapeReshapemel_stft/transpose_2:y:0mel_stft/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mel_stft/Reshape´
#mel_stft/transpose_3/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02%
#mel_stft/transpose_3/ReadVariableOp
mel_stft/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
mel_stft/transpose_3/perm´
mel_stft/transpose_3	Transpose+mel_stft/transpose_3/ReadVariableOp:value:0"mel_stft/transpose_3/perm:output:0*
T0*
_output_shapes
:	P2
mel_stft/transpose_3
mel_stft/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ÿÿÿÿ2
mel_stft/Reshape_1/shape
mel_stft/Reshape_1Reshapemel_stft/transpose_3:y:0!mel_stft/Reshape_1/shape:output:0*
T0*
_output_shapes
:	P2
mel_stft/Reshape_1
mel_stft/MatMulMatMulmel_stft/Reshape:output:0mel_stft/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP2
mel_stft/MatMulz
mel_stft/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/Reshape_2/shape/1z
mel_stft/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}2
mel_stft/Reshape_2/shape/2z
mel_stft/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :P2
mel_stft/Reshape_2/shape/3ô
mel_stft/Reshape_2/shapePackmel_stft/unstack:output:0#mel_stft/Reshape_2/shape/1:output:0#mel_stft/Reshape_2/shape/2:output:0#mel_stft/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2
mel_stft/Reshape_2/shape«
mel_stft/Reshape_2Reshapemel_stft/MatMul:product:0!mel_stft/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P2
mel_stft/Reshape_2
mel_stft/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_4/perm´
mel_stft/transpose_4	Transposemel_stft/Reshape_2:output:0"mel_stft/transpose_4/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/transpose_4e
mel_stft/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mel_stft/Consti
mel_stft/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
mel_stft/Const_1º
mel_stft/clip_by_value/MinimumMinimummel_stft/transpose_4:y:0mel_stft/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
mel_stft/clip_by_value/Minimum²
mel_stft/clip_by_valueMaximum"mel_stft/clip_by_value/Minimum:z:0mel_stft/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/clip_by_value|
mel_stft/SqrtSqrtmel_stft/clip_by_value:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Sqrti
mel_stft/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
mel_stft/Pow_2/y
mel_stft/Pow_2Powmel_stft/Sqrt:y:0mel_stft/Pow_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Pow_2m
mel_stft/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
mel_stft/Maximum/y
mel_stft/MaximumMaximummel_stft/Pow_2:z:0mel_stft/Maximum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximums
mel_stft/LogLogmel_stft/Maximum:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Loge
mel_stft/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mel_stft/mul/x
mel_stft/mulMulmel_stft/mul/x:output:0mel_stft/Log:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/mulm
mel_stft/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *]@2
mel_stft/truediv/y
mel_stft/truedivRealDivmel_stft/mul:z:0mel_stft/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/truediv
mel_stft/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/Max/reduction_indices­
mel_stft/MaxMaxmel_stft/truediv:z:0'mel_stft/Max/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
mel_stft/Max
mel_stft/subSubmel_stft/truediv:z:0mel_stft/Max:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/subq
mel_stft/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â2
mel_stft/Maximum_1/y
mel_stft/Maximum_1Maximummel_stft/sub:z:0mel_stft/Maximum_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximum_1©
(normalization2d_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2*
(normalization2d_7/Mean/reduction_indicesÎ
normalization2d_7/MeanMeanmel_stft/Maximum_1:z:01normalization2d_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
normalization2d_7/Meanß
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2E
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices
1normalization2d_7/reduce_std/reduce_variance/MeanMeanmel_stft/Maximum_1:z:0Lnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(23
1normalization2d_7/reduce_std/reduce_variance/Meanù
0normalization2d_7/reduce_std/reduce_variance/subSubmel_stft/Maximum_1:z:0:normalization2d_7/reduce_std/reduce_variance/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}22
0normalization2d_7/reduce_std/reduce_variance/subä
3normalization2d_7/reduce_std/reduce_variance/SquareSquare4normalization2d_7/reduce_std/reduce_variance/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}25
3normalization2d_7/reduce_std/reduce_variance/Squareã
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2G
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesÆ
3normalization2d_7/reduce_std/reduce_variance/Mean_1Mean7normalization2d_7/reduce_std/reduce_variance/Square:y:0Nnormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(25
3normalization2d_7/reduce_std/reduce_variance/Mean_1Æ
!normalization2d_7/reduce_std/SqrtSqrt<normalization2d_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!normalization2d_7/reduce_std/Sqrt¨
normalization2d_7/subSubmel_stft/Maximum_1:z:0normalization2d_7/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/subw
normalization2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
normalization2d_7/add/yº
normalization2d_7/addAddV2%normalization2d_7/reduce_std/Sqrt:y:0 normalization2d_7/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization2d_7/add±
normalization2d_7/truedivRealDivnormalization2d_7/sub:z:0normalization2d_7/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/truediv´
squeeze_last_dim/SqueezeSqueezenormalization2d_7/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
squeeze_last_dim/Squeeze{
simple_rnn_13/ShapeShape!squeeze_last_dim/Squeeze:output:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape
!simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_13/strided_slice/stack
#simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_1
#simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_2¶
simple_rnn_13/strided_sliceStridedSlicesimple_rnn_13/Shape:output:0*simple_rnn_13/strided_slice/stack:output:0,simple_rnn_13/strided_slice/stack_1:output:0,simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slicex
simple_rnn_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/mul/y¤
simple_rnn_13/zeros/mulMul$simple_rnn_13/strided_slice:output:0"simple_rnn_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/mul{
simple_rnn_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
simple_rnn_13/zeros/Less/y
simple_rnn_13/zeros/LessLesssimple_rnn_13/zeros/mul:z:0#simple_rnn_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/Less~
simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/packed/1»
simple_rnn_13/zeros/packedPack$simple_rnn_13/strided_slice:output:0%simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_13/zeros/packed{
simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_13/zeros/Const­
simple_rnn_13/zerosFill#simple_rnn_13/zeros/packed:output:0"simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_13/zeros
simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_13/transpose/perm¿
simple_rnn_13/transpose	Transpose!squeeze_last_dim/Squeeze:output:0%simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
simple_rnn_13/transposey
simple_rnn_13/Shape_1Shapesimple_rnn_13/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape_1
#simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_1/stack
%simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_1
%simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_2Â
simple_rnn_13/strided_slice_1StridedSlicesimple_rnn_13/Shape_1:output:0,simple_rnn_13/strided_slice_1/stack:output:0.simple_rnn_13/strided_slice_1/stack_1:output:0.simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slice_1¡
)simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)simple_rnn_13/TensorArrayV2/element_shapeê
simple_rnn_13/TensorArrayV2TensorListReserve2simple_rnn_13/TensorArrayV2/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2Û
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2E
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape°
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_13/transpose:y:0Lsimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensor
#simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_2/stack
%simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_1
%simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_2Ð
simple_rnn_13/strided_slice_2StridedSlicesimple_rnn_13/transpose:y:0,simple_rnn_13/strided_slice_2/stack:output:0.simple_rnn_13/strided_slice_2/stack_1:output:0.simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
simple_rnn_13/strided_slice_2ð
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype028
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpö
'simple_rnn_13/simple_rnn_cell_13/MatMulMatMul&simple_rnn_13/strided_slice_2:output:0>simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'simple_rnn_13/simple_rnn_cell_13/MatMulï
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp
(simple_rnn_13/simple_rnn_cell_13/BiasAddBiasAdd1simple_rnn_13/simple_rnn_cell_13/MatMul:product:0?simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(simple_rnn_13/simple_rnn_cell_13/BiasAddö
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpò
)simple_rnn_13/simple_rnn_cell_13/MatMul_1MatMulsimple_rnn_13/zeros:output:0@simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)simple_rnn_13/simple_rnn_cell_13/MatMul_1ï
$simple_rnn_13/simple_rnn_cell_13/addAddV21simple_rnn_13/simple_rnn_cell_13/BiasAdd:output:03simple_rnn_13/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$simple_rnn_13/simple_rnn_cell_13/add²
%simple_rnn_13/simple_rnn_cell_13/TanhTanh(simple_rnn_13/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%simple_rnn_13/simple_rnn_cell_13/Tanh«
+simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2-
+simple_rnn_13/TensorArrayV2_1/element_shapeð
simple_rnn_13/TensorArrayV2_1TensorListReserve4simple_rnn_13/TensorArrayV2_1/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2_1j
simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_13/time
&simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&simple_rnn_13/while/maximum_iterations
 simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_13/while/loop_counter
simple_rnn_13/whileWhile)simple_rnn_13/while/loop_counter:output:0/simple_rnn_13/while/maximum_iterations:output:0simple_rnn_13/time:output:0&simple_rnn_13/TensorArrayV2_1:handle:0simple_rnn_13/zeros:output:0&simple_rnn_13/strided_slice_1:output:0Esimple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_13_while_body_480664*+
cond#R!
simple_rnn_13_while_cond_480663*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
simple_rnn_13/whileÑ
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2@
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape 
0simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_13/while:output:3Gsimple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype022
0simple_rnn_13/TensorArrayV2Stack/TensorListStack
#simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2%
#simple_rnn_13/strided_slice_3/stack
%simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_13/strided_slice_3/stack_1
%simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_3/stack_2î
simple_rnn_13/strided_slice_3StridedSlice9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_13/strided_slice_3/stack:output:0.simple_rnn_13/strided_slice_3/stack_1:output:0.simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
simple_rnn_13/strided_slice_3
simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_13/transpose_1/permÝ
simple_rnn_13/transpose_1	Transpose9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
simple_rnn_13/transpose_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_8/Const
flatten_8/ReshapeReshapesimple_rnn_13/transpose_1:y:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
flatten_8/Reshape©
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype02 
dense_37/MatMul/ReadVariableOp¢
dense_37/MatMulMatMulflatten_8/Reshape:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/MatMul§
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp¥
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/BiasAdds
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/Relu¨
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_38/MatMul/ReadVariableOp£
dense_38/MatMulMatMuldense_37/Relu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/MatMul§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp¥
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/BiasAdds
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Relu¨
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

: $*
dtype02 
dense_39/MatMul/ReadVariableOp£
dense_39/MatMulMatMuldense_38/Relu:activations:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/Softmax
IdentityIdentitydense_39/Softmax:softmax:0^simple_rnn_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::2*
simple_rnn_13/whilesimple_rnn_13/while:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
ØÙ
ÿ
I__inference_sequential_19_layer_call_and_return_conditional_losses_481261
reshape_13_input0
,mel_stft_convolution_readvariableop_resource2
.mel_stft_convolution_1_readvariableop_resource,
(mel_stft_shape_1_readvariableop_resourceC
?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resourceD
@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceE
Asimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource
identity¢simple_rnn_13/whiled
reshape_13/ShapeShapereshape_13_input*
T0*
_output_shapes
:2
reshape_13/Shape
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2¤
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_13/Reshape/shape/2×
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape
reshape_13/ReshapeReshapereshape_13_input!reshape_13/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
reshape_13/Reshape
mel_stft/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
mel_stft/strided_slice/stack
mel_stft/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
mel_stft/strided_slice/stack_1
mel_stft/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/strided_slice/stack_2¼
mel_stft/strided_sliceStridedSlicereshape_13/Reshape:output:0%mel_stft/strided_slice/stack:output:0'mel_stft/strided_slice/stack_1:output:0'mel_stft/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask2
mel_stft/strided_slice
mel_stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mel_stft/transpose/perm¯
mel_stft/transpose	Transposemel_stft/strided_slice:output:0 mel_stft/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transposet
mel_stft/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/ExpandDims/dim­
mel_stft/ExpandDims
ExpandDimsmel_stft/transpose:y:0 mel_stft/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/ExpandDimsÁ
#mel_stft/convolution/ReadVariableOpReadVariableOp,mel_stft_convolution_readvariableop_resource*(
_output_shapes
:*
dtype02%
#mel_stft/convolution/ReadVariableOpå
mel_stft/convolutionConv2Dmel_stft/ExpandDims:output:0+mel_stft/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolutionÇ
%mel_stft/convolution_1/ReadVariableOpReadVariableOp.mel_stft_convolution_1_readvariableop_resource*(
_output_shapes
:*
dtype02'
%mel_stft/convolution_1/ReadVariableOpë
mel_stft/convolution_1Conv2Dmel_stft/ExpandDims:output:0-mel_stft/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolution_1e
mel_stft/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow/y
mel_stft/powPowmel_stft/convolution:output:0mel_stft/pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/powi
mel_stft/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow_1/y
mel_stft/pow_1Powmel_stft/convolution_1:output:0mel_stft/pow_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/pow_1
mel_stft/addAddV2mel_stft/pow:z:0mel_stft/pow_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/add
mel_stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_1/permª
mel_stft/transpose_1	Transposemel_stft/add:z:0"mel_stft/transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_1
mel_stft/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_2/perm²
mel_stft/transpose_2	Transposemel_stft/transpose_1:y:0"mel_stft/transpose_2/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_2h
mel_stft/ShapeShapemel_stft/transpose_2:y:0*
T0*
_output_shapes
:2
mel_stft/Shapey
mel_stft/unstackUnpackmel_stft/Shape:output:0*
T0*
_output_shapes

: : : : *	
num2
mel_stft/unstack¬
mel_stft/Shape_1/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02!
mel_stft/Shape_1/ReadVariableOpu
mel_stft/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"  P   2
mel_stft/Shape_1{
mel_stft/unstack_1Unpackmel_stft/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
mel_stft/unstack_1
mel_stft/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
mel_stft/Reshape/shape
mel_stft/ReshapeReshapemel_stft/transpose_2:y:0mel_stft/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mel_stft/Reshape´
#mel_stft/transpose_3/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02%
#mel_stft/transpose_3/ReadVariableOp
mel_stft/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
mel_stft/transpose_3/perm´
mel_stft/transpose_3	Transpose+mel_stft/transpose_3/ReadVariableOp:value:0"mel_stft/transpose_3/perm:output:0*
T0*
_output_shapes
:	P2
mel_stft/transpose_3
mel_stft/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ÿÿÿÿ2
mel_stft/Reshape_1/shape
mel_stft/Reshape_1Reshapemel_stft/transpose_3:y:0!mel_stft/Reshape_1/shape:output:0*
T0*
_output_shapes
:	P2
mel_stft/Reshape_1
mel_stft/MatMulMatMulmel_stft/Reshape:output:0mel_stft/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP2
mel_stft/MatMulz
mel_stft/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/Reshape_2/shape/1z
mel_stft/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}2
mel_stft/Reshape_2/shape/2z
mel_stft/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :P2
mel_stft/Reshape_2/shape/3ô
mel_stft/Reshape_2/shapePackmel_stft/unstack:output:0#mel_stft/Reshape_2/shape/1:output:0#mel_stft/Reshape_2/shape/2:output:0#mel_stft/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2
mel_stft/Reshape_2/shape«
mel_stft/Reshape_2Reshapemel_stft/MatMul:product:0!mel_stft/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P2
mel_stft/Reshape_2
mel_stft/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_4/perm´
mel_stft/transpose_4	Transposemel_stft/Reshape_2:output:0"mel_stft/transpose_4/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/transpose_4e
mel_stft/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mel_stft/Consti
mel_stft/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
mel_stft/Const_1º
mel_stft/clip_by_value/MinimumMinimummel_stft/transpose_4:y:0mel_stft/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
mel_stft/clip_by_value/Minimum²
mel_stft/clip_by_valueMaximum"mel_stft/clip_by_value/Minimum:z:0mel_stft/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/clip_by_value|
mel_stft/SqrtSqrtmel_stft/clip_by_value:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Sqrti
mel_stft/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
mel_stft/Pow_2/y
mel_stft/Pow_2Powmel_stft/Sqrt:y:0mel_stft/Pow_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Pow_2m
mel_stft/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
mel_stft/Maximum/y
mel_stft/MaximumMaximummel_stft/Pow_2:z:0mel_stft/Maximum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximums
mel_stft/LogLogmel_stft/Maximum:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Loge
mel_stft/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mel_stft/mul/x
mel_stft/mulMulmel_stft/mul/x:output:0mel_stft/Log:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/mulm
mel_stft/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *]@2
mel_stft/truediv/y
mel_stft/truedivRealDivmel_stft/mul:z:0mel_stft/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/truediv
mel_stft/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/Max/reduction_indices­
mel_stft/MaxMaxmel_stft/truediv:z:0'mel_stft/Max/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
mel_stft/Max
mel_stft/subSubmel_stft/truediv:z:0mel_stft/Max:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/subq
mel_stft/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â2
mel_stft/Maximum_1/y
mel_stft/Maximum_1Maximummel_stft/sub:z:0mel_stft/Maximum_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximum_1©
(normalization2d_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2*
(normalization2d_7/Mean/reduction_indicesÎ
normalization2d_7/MeanMeanmel_stft/Maximum_1:z:01normalization2d_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
normalization2d_7/Meanß
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2E
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices
1normalization2d_7/reduce_std/reduce_variance/MeanMeanmel_stft/Maximum_1:z:0Lnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(23
1normalization2d_7/reduce_std/reduce_variance/Meanù
0normalization2d_7/reduce_std/reduce_variance/subSubmel_stft/Maximum_1:z:0:normalization2d_7/reduce_std/reduce_variance/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}22
0normalization2d_7/reduce_std/reduce_variance/subä
3normalization2d_7/reduce_std/reduce_variance/SquareSquare4normalization2d_7/reduce_std/reduce_variance/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}25
3normalization2d_7/reduce_std/reduce_variance/Squareã
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2G
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesÆ
3normalization2d_7/reduce_std/reduce_variance/Mean_1Mean7normalization2d_7/reduce_std/reduce_variance/Square:y:0Nnormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(25
3normalization2d_7/reduce_std/reduce_variance/Mean_1Æ
!normalization2d_7/reduce_std/SqrtSqrt<normalization2d_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!normalization2d_7/reduce_std/Sqrt¨
normalization2d_7/subSubmel_stft/Maximum_1:z:0normalization2d_7/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/subw
normalization2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
normalization2d_7/add/yº
normalization2d_7/addAddV2%normalization2d_7/reduce_std/Sqrt:y:0 normalization2d_7/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization2d_7/add±
normalization2d_7/truedivRealDivnormalization2d_7/sub:z:0normalization2d_7/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/truediv´
squeeze_last_dim/SqueezeSqueezenormalization2d_7/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
squeeze_last_dim/Squeeze{
simple_rnn_13/ShapeShape!squeeze_last_dim/Squeeze:output:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape
!simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_13/strided_slice/stack
#simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_1
#simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_2¶
simple_rnn_13/strided_sliceStridedSlicesimple_rnn_13/Shape:output:0*simple_rnn_13/strided_slice/stack:output:0,simple_rnn_13/strided_slice/stack_1:output:0,simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slicex
simple_rnn_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/mul/y¤
simple_rnn_13/zeros/mulMul$simple_rnn_13/strided_slice:output:0"simple_rnn_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/mul{
simple_rnn_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
simple_rnn_13/zeros/Less/y
simple_rnn_13/zeros/LessLesssimple_rnn_13/zeros/mul:z:0#simple_rnn_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/Less~
simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/packed/1»
simple_rnn_13/zeros/packedPack$simple_rnn_13/strided_slice:output:0%simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_13/zeros/packed{
simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_13/zeros/Const­
simple_rnn_13/zerosFill#simple_rnn_13/zeros/packed:output:0"simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_13/zeros
simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_13/transpose/perm¿
simple_rnn_13/transpose	Transpose!squeeze_last_dim/Squeeze:output:0%simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
simple_rnn_13/transposey
simple_rnn_13/Shape_1Shapesimple_rnn_13/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape_1
#simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_1/stack
%simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_1
%simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_2Â
simple_rnn_13/strided_slice_1StridedSlicesimple_rnn_13/Shape_1:output:0,simple_rnn_13/strided_slice_1/stack:output:0.simple_rnn_13/strided_slice_1/stack_1:output:0.simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slice_1¡
)simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)simple_rnn_13/TensorArrayV2/element_shapeê
simple_rnn_13/TensorArrayV2TensorListReserve2simple_rnn_13/TensorArrayV2/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2Û
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2E
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape°
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_13/transpose:y:0Lsimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensor
#simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_2/stack
%simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_1
%simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_2Ð
simple_rnn_13/strided_slice_2StridedSlicesimple_rnn_13/transpose:y:0,simple_rnn_13/strided_slice_2/stack:output:0.simple_rnn_13/strided_slice_2/stack_1:output:0.simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
simple_rnn_13/strided_slice_2ð
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype028
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpö
'simple_rnn_13/simple_rnn_cell_13/MatMulMatMul&simple_rnn_13/strided_slice_2:output:0>simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'simple_rnn_13/simple_rnn_cell_13/MatMulï
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp
(simple_rnn_13/simple_rnn_cell_13/BiasAddBiasAdd1simple_rnn_13/simple_rnn_cell_13/MatMul:product:0?simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(simple_rnn_13/simple_rnn_cell_13/BiasAddö
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpò
)simple_rnn_13/simple_rnn_cell_13/MatMul_1MatMulsimple_rnn_13/zeros:output:0@simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)simple_rnn_13/simple_rnn_cell_13/MatMul_1ï
$simple_rnn_13/simple_rnn_cell_13/addAddV21simple_rnn_13/simple_rnn_cell_13/BiasAdd:output:03simple_rnn_13/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$simple_rnn_13/simple_rnn_cell_13/add²
%simple_rnn_13/simple_rnn_cell_13/TanhTanh(simple_rnn_13/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%simple_rnn_13/simple_rnn_cell_13/Tanh«
+simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2-
+simple_rnn_13/TensorArrayV2_1/element_shapeð
simple_rnn_13/TensorArrayV2_1TensorListReserve4simple_rnn_13/TensorArrayV2_1/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2_1j
simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_13/time
&simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&simple_rnn_13/while/maximum_iterations
 simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_13/while/loop_counter
simple_rnn_13/whileWhile)simple_rnn_13/while/loop_counter:output:0/simple_rnn_13/while/maximum_iterations:output:0simple_rnn_13/time:output:0&simple_rnn_13/TensorArrayV2_1:handle:0simple_rnn_13/zeros:output:0&simple_rnn_13/strided_slice_1:output:0Esimple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_13_while_body_481172*+
cond#R!
simple_rnn_13_while_cond_481171*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
simple_rnn_13/whileÑ
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2@
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape 
0simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_13/while:output:3Gsimple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype022
0simple_rnn_13/TensorArrayV2Stack/TensorListStack
#simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2%
#simple_rnn_13/strided_slice_3/stack
%simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_13/strided_slice_3/stack_1
%simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_3/stack_2î
simple_rnn_13/strided_slice_3StridedSlice9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_13/strided_slice_3/stack:output:0.simple_rnn_13/strided_slice_3/stack_1:output:0.simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
simple_rnn_13/strided_slice_3
simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_13/transpose_1/permÝ
simple_rnn_13/transpose_1	Transpose9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
simple_rnn_13/transpose_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_8/Const
flatten_8/ReshapeReshapesimple_rnn_13/transpose_1:y:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
flatten_8/Reshape©
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype02 
dense_37/MatMul/ReadVariableOp¢
dense_37/MatMulMatMulflatten_8/Reshape:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/MatMul§
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp¥
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/BiasAdds
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/Relu¨
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_38/MatMul/ReadVariableOp£
dense_38/MatMulMatMuldense_37/Relu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/MatMul§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp¥
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/BiasAdds
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Relu¨
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

: $*
dtype02 
dense_39/MatMul/ReadVariableOp£
dense_39/MatMulMatMuldense_38/Relu:activations:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/Softmax
IdentityIdentitydense_39/Softmax:softmax:0^simple_rnn_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::2*
simple_rnn_13/whilesimple_rnn_13/while:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
*
_user_specified_namereshape_13_input
8
é	
simple_rnn_13_while_body_4813978
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_27
3simple_rnn_13_while_simple_rnn_13_strided_slice_1_0s
osimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0L
Hsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0M
Isimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0 
simple_rnn_13_while_identity"
simple_rnn_13_while_identity_1"
simple_rnn_13_while_identity_2"
simple_rnn_13_while_identity_3"
simple_rnn_13_while_identity_45
1simple_rnn_13_while_simple_rnn_13_strided_slice_1q
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceJ
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceK
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceß
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2G
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape§
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_13_while_placeholderNsimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype029
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype02>
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp 
-simple_rnn_13/while/simple_rnn_cell_13/MatMulMatMul>simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2/
-simple_rnn_13/while/simple_rnn_cell_13/MatMul
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02?
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp
.simple_rnn_13/while/simple_rnn_cell_13/BiasAddBiasAdd7simple_rnn_13/while/simple_rnn_cell_13/MatMul:product:0Esimple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.simple_rnn_13/while/simple_rnn_cell_13/BiasAdd
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02@
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1MatMul!simple_rnn_13_while_placeholder_2Fsimple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@21
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1
*simple_rnn_13/while/simple_rnn_cell_13/addAddV27simple_rnn_13/while/simple_rnn_cell_13/BiasAdd:output:09simple_rnn_13/while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*simple_rnn_13/while/simple_rnn_cell_13/addÄ
+simple_rnn_13/while/simple_rnn_cell_13/TanhTanh.simple_rnn_13/while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+simple_rnn_13/while/simple_rnn_cell_13/Tanh«
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_13_while_placeholder_1simple_rnn_13_while_placeholder/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add/y¡
simple_rnn_13/while/addAddV2simple_rnn_13_while_placeholder"simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add|
simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add_1/y¼
simple_rnn_13/while/add_1AddV24simple_rnn_13_while_simple_rnn_13_while_loop_counter$simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add_1
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/Identity©
simple_rnn_13/while/Identity_1Identity:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_1
simple_rnn_13/while/Identity_2Identitysimple_rnn_13/while/add:z:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_2·
simple_rnn_13/while/Identity_3IdentityHsimple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_3¯
simple_rnn_13/while/Identity_4Identity/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
simple_rnn_13/while/Identity_4"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0"I
simple_rnn_13_while_identity_1'simple_rnn_13/while/Identity_1:output:0"I
simple_rnn_13_while_identity_2'simple_rnn_13/while/Identity_2:output:0"I
simple_rnn_13_while_identity_3'simple_rnn_13/while/Identity_3:output:0"I
simple_rnn_13_while_identity_4'simple_rnn_13/while/Identity_4:output:0"h
1simple_rnn_13_while_simple_rnn_13_strided_slice_13simple_rnn_13_while_simple_rnn_13_strided_slice_1_0"
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0"à
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ð
ª
while_cond_481731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_481731___redundant_placeholder04
0while_while_cond_481731___redundant_placeholder14
0while_while_cond_481731___redundant_placeholder24
0while_while_cond_481731___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ï	
µ
3__inference_simple_rnn_cell_13_layer_call_fn_482311

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_4793602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ}:ÿÿÿÿÿÿÿÿÿ@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
°Ù
õ
I__inference_sequential_19_layer_call_and_return_conditional_losses_480978

inputs0
,mel_stft_convolution_readvariableop_resource2
.mel_stft_convolution_1_readvariableop_resource,
(mel_stft_shape_1_readvariableop_resourceC
?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resourceD
@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceE
Asimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource
identity¢simple_rnn_13/whileZ
reshape_13/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_13/Shape
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2¤
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_13/Reshape/shape/2×
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape
reshape_13/ReshapeReshapeinputs!reshape_13/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
reshape_13/Reshape
mel_stft/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
mel_stft/strided_slice/stack
mel_stft/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
mel_stft/strided_slice/stack_1
mel_stft/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/strided_slice/stack_2¼
mel_stft/strided_sliceStridedSlicereshape_13/Reshape:output:0%mel_stft/strided_slice/stack:output:0'mel_stft/strided_slice/stack_1:output:0'mel_stft/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask2
mel_stft/strided_slice
mel_stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mel_stft/transpose/perm¯
mel_stft/transpose	Transposemel_stft/strided_slice:output:0 mel_stft/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transposet
mel_stft/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/ExpandDims/dim­
mel_stft/ExpandDims
ExpandDimsmel_stft/transpose:y:0 mel_stft/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/ExpandDimsÁ
#mel_stft/convolution/ReadVariableOpReadVariableOp,mel_stft_convolution_readvariableop_resource*(
_output_shapes
:*
dtype02%
#mel_stft/convolution/ReadVariableOpå
mel_stft/convolutionConv2Dmel_stft/ExpandDims:output:0+mel_stft/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolutionÇ
%mel_stft/convolution_1/ReadVariableOpReadVariableOp.mel_stft_convolution_1_readvariableop_resource*(
_output_shapes
:*
dtype02'
%mel_stft/convolution_1/ReadVariableOpë
mel_stft/convolution_1Conv2Dmel_stft/ExpandDims:output:0-mel_stft/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
mel_stft/convolution_1e
mel_stft/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow/y
mel_stft/powPowmel_stft/convolution:output:0mel_stft/pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/powi
mel_stft/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mel_stft/pow_1/y
mel_stft/pow_1Powmel_stft/convolution_1:output:0mel_stft/pow_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/pow_1
mel_stft/addAddV2mel_stft/pow:z:0mel_stft/pow_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/add
mel_stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_1/permª
mel_stft/transpose_1	Transposemel_stft/add:z:0"mel_stft/transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_1
mel_stft/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_2/perm²
mel_stft/transpose_2	Transposemel_stft/transpose_1:y:0"mel_stft/transpose_2/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
mel_stft/transpose_2h
mel_stft/ShapeShapemel_stft/transpose_2:y:0*
T0*
_output_shapes
:2
mel_stft/Shapey
mel_stft/unstackUnpackmel_stft/Shape:output:0*
T0*
_output_shapes

: : : : *	
num2
mel_stft/unstack¬
mel_stft/Shape_1/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02!
mel_stft/Shape_1/ReadVariableOpu
mel_stft/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"  P   2
mel_stft/Shape_1{
mel_stft/unstack_1Unpackmel_stft/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
mel_stft/unstack_1
mel_stft/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
mel_stft/Reshape/shape
mel_stft/ReshapeReshapemel_stft/transpose_2:y:0mel_stft/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mel_stft/Reshape´
#mel_stft/transpose_3/ReadVariableOpReadVariableOp(mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02%
#mel_stft/transpose_3/ReadVariableOp
mel_stft/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
mel_stft/transpose_3/perm´
mel_stft/transpose_3	Transpose+mel_stft/transpose_3/ReadVariableOp:value:0"mel_stft/transpose_3/perm:output:0*
T0*
_output_shapes
:	P2
mel_stft/transpose_3
mel_stft/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ÿÿÿÿ2
mel_stft/Reshape_1/shape
mel_stft/Reshape_1Reshapemel_stft/transpose_3:y:0!mel_stft/Reshape_1/shape:output:0*
T0*
_output_shapes
:	P2
mel_stft/Reshape_1
mel_stft/MatMulMatMulmel_stft/Reshape:output:0mel_stft/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP2
mel_stft/MatMulz
mel_stft/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
mel_stft/Reshape_2/shape/1z
mel_stft/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}2
mel_stft/Reshape_2/shape/2z
mel_stft/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :P2
mel_stft/Reshape_2/shape/3ô
mel_stft/Reshape_2/shapePackmel_stft/unstack:output:0#mel_stft/Reshape_2/shape/1:output:0#mel_stft/Reshape_2/shape/2:output:0#mel_stft/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2
mel_stft/Reshape_2/shape«
mel_stft/Reshape_2Reshapemel_stft/MatMul:product:0!mel_stft/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P2
mel_stft/Reshape_2
mel_stft/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
mel_stft/transpose_4/perm´
mel_stft/transpose_4	Transposemel_stft/Reshape_2:output:0"mel_stft/transpose_4/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/transpose_4e
mel_stft/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mel_stft/Consti
mel_stft/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
mel_stft/Const_1º
mel_stft/clip_by_value/MinimumMinimummel_stft/transpose_4:y:0mel_stft/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
mel_stft/clip_by_value/Minimum²
mel_stft/clip_by_valueMaximum"mel_stft/clip_by_value/Minimum:z:0mel_stft/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/clip_by_value|
mel_stft/SqrtSqrtmel_stft/clip_by_value:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Sqrti
mel_stft/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
mel_stft/Pow_2/y
mel_stft/Pow_2Powmel_stft/Sqrt:y:0mel_stft/Pow_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Pow_2m
mel_stft/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
mel_stft/Maximum/y
mel_stft/MaximumMaximummel_stft/Pow_2:z:0mel_stft/Maximum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximums
mel_stft/LogLogmel_stft/Maximum:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Loge
mel_stft/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mel_stft/mul/x
mel_stft/mulMulmel_stft/mul/x:output:0mel_stft/Log:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/mulm
mel_stft/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *]@2
mel_stft/truediv/y
mel_stft/truedivRealDivmel_stft/mul:z:0mel_stft/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/truediv
mel_stft/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2 
mel_stft/Max/reduction_indices­
mel_stft/MaxMaxmel_stft/truediv:z:0'mel_stft/Max/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
mel_stft/Max
mel_stft/subSubmel_stft/truediv:z:0mel_stft/Max:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/subq
mel_stft/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â2
mel_stft/Maximum_1/y
mel_stft/Maximum_1Maximummel_stft/sub:z:0mel_stft/Maximum_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mel_stft/Maximum_1©
(normalization2d_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2*
(normalization2d_7/Mean/reduction_indicesÎ
normalization2d_7/MeanMeanmel_stft/Maximum_1:z:01normalization2d_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
normalization2d_7/Meanß
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2E
Cnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices
1normalization2d_7/reduce_std/reduce_variance/MeanMeanmel_stft/Maximum_1:z:0Lnormalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(23
1normalization2d_7/reduce_std/reduce_variance/Meanù
0normalization2d_7/reduce_std/reduce_variance/subSubmel_stft/Maximum_1:z:0:normalization2d_7/reduce_std/reduce_variance/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}22
0normalization2d_7/reduce_std/reduce_variance/subä
3normalization2d_7/reduce_std/reduce_variance/SquareSquare4normalization2d_7/reduce_std/reduce_variance/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}25
3normalization2d_7/reduce_std/reduce_variance/Squareã
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2G
Enormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesÆ
3normalization2d_7/reduce_std/reduce_variance/Mean_1Mean7normalization2d_7/reduce_std/reduce_variance/Square:y:0Nnormalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(25
3normalization2d_7/reduce_std/reduce_variance/Mean_1Æ
!normalization2d_7/reduce_std/SqrtSqrt<normalization2d_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!normalization2d_7/reduce_std/Sqrt¨
normalization2d_7/subSubmel_stft/Maximum_1:z:0normalization2d_7/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/subw
normalization2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
normalization2d_7/add/yº
normalization2d_7/addAddV2%normalization2d_7/reduce_std/Sqrt:y:0 normalization2d_7/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization2d_7/add±
normalization2d_7/truedivRealDivnormalization2d_7/sub:z:0normalization2d_7/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
normalization2d_7/truediv´
squeeze_last_dim/SqueezeSqueezenormalization2d_7/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
squeeze_last_dim/Squeeze{
simple_rnn_13/ShapeShape!squeeze_last_dim/Squeeze:output:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape
!simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_13/strided_slice/stack
#simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_1
#simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_13/strided_slice/stack_2¶
simple_rnn_13/strided_sliceStridedSlicesimple_rnn_13/Shape:output:0*simple_rnn_13/strided_slice/stack:output:0,simple_rnn_13/strided_slice/stack_1:output:0,simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slicex
simple_rnn_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/mul/y¤
simple_rnn_13/zeros/mulMul$simple_rnn_13/strided_slice:output:0"simple_rnn_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/mul{
simple_rnn_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
simple_rnn_13/zeros/Less/y
simple_rnn_13/zeros/LessLesssimple_rnn_13/zeros/mul:z:0#simple_rnn_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/zeros/Less~
simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
simple_rnn_13/zeros/packed/1»
simple_rnn_13/zeros/packedPack$simple_rnn_13/strided_slice:output:0%simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_13/zeros/packed{
simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_13/zeros/Const­
simple_rnn_13/zerosFill#simple_rnn_13/zeros/packed:output:0"simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_13/zeros
simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_13/transpose/perm¿
simple_rnn_13/transpose	Transpose!squeeze_last_dim/Squeeze:output:0%simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
simple_rnn_13/transposey
simple_rnn_13/Shape_1Shapesimple_rnn_13/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_13/Shape_1
#simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_1/stack
%simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_1
%simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_1/stack_2Â
simple_rnn_13/strided_slice_1StridedSlicesimple_rnn_13/Shape_1:output:0,simple_rnn_13/strided_slice_1/stack:output:0.simple_rnn_13/strided_slice_1/stack_1:output:0.simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_13/strided_slice_1¡
)simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)simple_rnn_13/TensorArrayV2/element_shapeê
simple_rnn_13/TensorArrayV2TensorListReserve2simple_rnn_13/TensorArrayV2/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2Û
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2E
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape°
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_13/transpose:y:0Lsimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensor
#simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_13/strided_slice_2/stack
%simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_1
%simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_2/stack_2Ð
simple_rnn_13/strided_slice_2StridedSlicesimple_rnn_13/transpose:y:0,simple_rnn_13/strided_slice_2/stack:output:0.simple_rnn_13/strided_slice_2/stack_1:output:0.simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
simple_rnn_13/strided_slice_2ð
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype028
6simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpö
'simple_rnn_13/simple_rnn_cell_13/MatMulMatMul&simple_rnn_13/strided_slice_2:output:0>simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'simple_rnn_13/simple_rnn_cell_13/MatMulï
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp
(simple_rnn_13/simple_rnn_cell_13/BiasAddBiasAdd1simple_rnn_13/simple_rnn_cell_13/MatMul:product:0?simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(simple_rnn_13/simple_rnn_cell_13/BiasAddö
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpò
)simple_rnn_13/simple_rnn_cell_13/MatMul_1MatMulsimple_rnn_13/zeros:output:0@simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)simple_rnn_13/simple_rnn_cell_13/MatMul_1ï
$simple_rnn_13/simple_rnn_cell_13/addAddV21simple_rnn_13/simple_rnn_cell_13/BiasAdd:output:03simple_rnn_13/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$simple_rnn_13/simple_rnn_cell_13/add²
%simple_rnn_13/simple_rnn_cell_13/TanhTanh(simple_rnn_13/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%simple_rnn_13/simple_rnn_cell_13/Tanh«
+simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2-
+simple_rnn_13/TensorArrayV2_1/element_shapeð
simple_rnn_13/TensorArrayV2_1TensorListReserve4simple_rnn_13/TensorArrayV2_1/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_13/TensorArrayV2_1j
simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_13/time
&simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&simple_rnn_13/while/maximum_iterations
 simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_13/while/loop_counter
simple_rnn_13/whileWhile)simple_rnn_13/while/loop_counter:output:0/simple_rnn_13/while/maximum_iterations:output:0simple_rnn_13/time:output:0&simple_rnn_13/TensorArrayV2_1:handle:0simple_rnn_13/zeros:output:0&simple_rnn_13/strided_slice_1:output:0Esimple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource@simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceAsimple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_13_while_body_480889*+
cond#R!
simple_rnn_13_while_cond_480888*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
simple_rnn_13/whileÑ
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2@
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape 
0simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_13/while:output:3Gsimple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype022
0simple_rnn_13/TensorArrayV2Stack/TensorListStack
#simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2%
#simple_rnn_13/strided_slice_3/stack
%simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_13/strided_slice_3/stack_1
%simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_13/strided_slice_3/stack_2î
simple_rnn_13/strided_slice_3StridedSlice9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_13/strided_slice_3/stack:output:0.simple_rnn_13/strided_slice_3/stack_1:output:0.simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
simple_rnn_13/strided_slice_3
simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_13/transpose_1/permÝ
simple_rnn_13/transpose_1	Transpose9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
simple_rnn_13/transpose_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_8/Const
flatten_8/ReshapeReshapesimple_rnn_13/transpose_1:y:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
flatten_8/Reshape©
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype02 
dense_37/MatMul/ReadVariableOp¢
dense_37/MatMulMatMulflatten_8/Reshape:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/MatMul§
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp¥
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/BiasAdds
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/Relu¨
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_38/MatMul/ReadVariableOp£
dense_38/MatMulMatMuldense_37/Relu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/MatMul§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp¥
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/BiasAdds
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Relu¨
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

: $*
dtype02 
dense_39/MatMul/ReadVariableOp£
dense_39/MatMulMatMuldense_38/Relu:activations:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
dense_39/Softmax
IdentityIdentitydense_39/Softmax:softmax:0^simple_rnn_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::2*
simple_rnn_13/whilesimple_rnn_13/while:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
ù
h
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_479952

inputs
identity{
SqueezeSqueezeinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
©)
§
I__inference_sequential_19_layer_call_and_return_conditional_losses_480462

inputs
mel_stft_480429
mel_stft_480431
mel_stft_480433
simple_rnn_13_480438
simple_rnn_13_480440
simple_rnn_13_480442
dense_37_480446
dense_37_480448
dense_38_480451
dense_38_480453
dense_39_480456
dense_39_480458
identity¢ dense_37/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ mel_stft/StatefulPartitionedCall¢%simple_rnn_13/StatefulPartitionedCallâ
reshape_13/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_13_layer_call_and_return_conditional_losses_4798232
reshape_13/PartitionedCallÏ
 mel_stft/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0mel_stft_480429mel_stft_480431mel_stft_480433*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mel_stft_layer_call_and_return_conditional_losses_4799022"
 mel_stft/StatefulPartitionedCall
!normalization2d_7/PartitionedCallPartitionedCall)mel_stft/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_normalization2d_7_layer_call_and_return_conditional_losses_4799392#
!normalization2d_7/PartitionedCall
 squeeze_last_dim/PartitionedCallPartitionedCall*normalization2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_4799572"
 squeeze_last_dim/PartitionedCallï
%simple_rnn_13/StatefulPartitionedCallStatefulPartitionedCall)squeeze_last_dim/PartitionedCall:output:0simple_rnn_13_480438simple_rnn_13_480440simple_rnn_13_480442*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_4801932'
%simple_rnn_13/StatefulPartitionedCall
flatten_8/PartitionedCallPartitionedCall.simple_rnn_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4802292
flatten_8/PartitionedCall³
 dense_37/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_37_480446dense_37_480448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4802482"
 dense_37/StatefulPartitionedCallº
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_480451dense_38_480453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_4802752"
 dense_38/StatefulPartitionedCallº
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_480456dense_39_480458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_4803022"
 dense_39/StatefulPartitionedCall±
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^mel_stft/StatefulPartitionedCall&^simple_rnn_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 mel_stft/StatefulPartitionedCall mel_stft/StatefulPartitionedCall2N
%simple_rnn_13/StatefulPartitionedCall%simple_rnn_13/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs


.__inference_simple_rnn_13_layer_call_fn_482178

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_4801932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿP}:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
¬
¬
D__inference_dense_37_layer_call_and_return_conditional_losses_482200

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ß
~
)__inference_dense_39_layer_call_fn_482249

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_4803022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ü<
ú
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_479797

inputs
simple_rnn_cell_13_479722
simple_rnn_cell_13_479724
simple_rnn_cell_13_479726
identity¢*simple_rnn_cell_13/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2
*simple_rnn_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_13_479722simple_rnn_cell_13_479724simple_rnn_cell_13_479726*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_4793602,
*simple_rnn_cell_13/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterü
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_13_479722simple_rnn_cell_13_479724simple_rnn_cell_13_479726*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_479734*
condR
while_cond_479733*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1¥
IdentityIdentitytranspose_1:y:0+^simple_rnn_cell_13/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}:::2X
*simple_rnn_cell_13/StatefulPartitionedCall*simple_rnn_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
Ä

.__inference_simple_rnn_13_layer_call_fn_481932
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_4797972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}
"
_user_specified_name
inputs/0
®	

.__inference_sequential_19_layer_call_fn_481036

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4804622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
Ì	
¤
.__inference_sequential_19_layer_call_fn_481544
reshape_13_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallreshape_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4804622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
*
_user_specified_namereshape_13_input
¡
F
*__inference_flatten_8_layer_call_fn_482189

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4802292
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@
 
_user_specified_nameinputs
Ð
ª
while_cond_479616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_479616___redundant_placeholder04
0while_while_cond_479616___redundant_placeholder14
0while_while_cond_479616___redundant_placeholder24
0while_while_cond_479616___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ùC

I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_480193

inputs5
1simple_rnn_cell_13_matmul_readvariableop_resource6
2simple_rnn_cell_13_biasadd_readvariableop_resource7
3simple_rnn_cell_13_matmul_1_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2Æ
(simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype02*
(simple_rnn_cell_13/MatMul/ReadVariableOp¾
simple_rnn_cell_13/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMulÅ
)simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)simple_rnn_cell_13/BiasAdd/ReadVariableOpÍ
simple_rnn_cell_13/BiasAddBiasAdd#simple_rnn_cell_13/MatMul:product:01simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/BiasAddÌ
*simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*simple_rnn_cell_13/MatMul_1/ReadVariableOpº
simple_rnn_cell_13/MatMul_1MatMulzeros:output:02simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMul_1·
simple_rnn_cell_13/addAddV2#simple_rnn_cell_13/BiasAdd:output:0%simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/add
simple_rnn_cell_13/TanhTanhsimple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_13_matmul_readvariableop_resource2simple_rnn_cell_13_biasadd_readvariableop_resource3simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_480127*
condR
while_cond_480126*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
transpose_1o
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿP}:::2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
µ#

while_body_479617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_13_479639_0%
!while_simple_rnn_cell_13_479641_0%
!while_simple_rnn_cell_13_479643_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_13_479639#
while_simple_rnn_cell_13_479641#
while_simple_rnn_cell_13_479643¢0while/simple_rnn_cell_13/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÞ
0while/simple_rnn_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_13_479639_0!while_simple_rnn_cell_13_479641_0!while_simple_rnn_cell_13_479643_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_47934322
0while/simple_rnn_cell_13/StatefulPartitionedCallý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity¤
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2À
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ð
while/Identity_4Identity9while/simple_rnn_cell_13/StatefulPartitionedCall:output:11^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_13_479639!while_simple_rnn_cell_13_479639_0"D
while_simple_rnn_cell_13_479641!while_simple_rnn_cell_13_479641_0"D
while_simple_rnn_cell_13_479643!while_simple_rnn_cell_13_479643_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::2d
0while/simple_rnn_cell_13/StatefulPartitionedCall0while/simple_rnn_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
¬
¬
D__inference_dense_37_layer_call_and_return_conditional_losses_480248

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs


.__inference_simple_rnn_13_layer_call_fn_482167

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_4800812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿP}:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
Í
ø
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_479360

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ}:ÿÿÿÿÿÿÿÿÿ@::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
µ
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_482184

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@
 
_user_specified_nameinputs
¸D

I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_481910
inputs_05
1simple_rnn_cell_13_matmul_readvariableop_resource6
2simple_rnn_cell_13_biasadd_readvariableop_resource7
3simple_rnn_cell_13_matmul_1_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2Æ
(simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype02*
(simple_rnn_cell_13/MatMul/ReadVariableOp¾
simple_rnn_cell_13/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMulÅ
)simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)simple_rnn_cell_13/BiasAdd/ReadVariableOpÍ
simple_rnn_cell_13/BiasAddBiasAdd#simple_rnn_cell_13/MatMul:product:01simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/BiasAddÌ
*simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*simple_rnn_cell_13/MatMul_1/ReadVariableOpº
simple_rnn_cell_13/MatMul_1MatMulzeros:output:02simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMul_1·
simple_rnn_cell_13/addAddV2#simple_rnn_cell_13/BiasAdd:output:0%simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/add
simple_rnn_cell_13/TanhTanhsimple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_13_matmul_readvariableop_resource2simple_rnn_cell_13_biasadd_readvariableop_resource3simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_481844*
condR
while_cond_481843*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1x
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}:::2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}
"
_user_specified_name
inputs/0
¥
G
+__inference_reshape_13_layer_call_fn_481562

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_13_layer_call_and_return_conditional_losses_4798232
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
¸
I
2__inference_normalization2d_7_layer_call_fn_481666
x
identityÑ
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_normalization2d_7_layer_call_and_return_conditional_losses_4799392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}

_user_specified_namex
Ó
ú
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_482283

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ}:ÿÿÿÿÿÿÿÿÿ@::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
ß
~
)__inference_dense_38_layer_call_fn_482229

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_4802752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è

´
simple_rnn_13_while_cond_4811718
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_2:
6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481171___redundant_placeholder0P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481171___redundant_placeholder1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481171___redundant_placeholder2P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481171___redundant_placeholder3 
simple_rnn_13_while_identity
¶
simple_rnn_13/while/LessLesssimple_rnn_13_while_placeholder6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_13/while/Less
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_13/while/Identity"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
è

´
simple_rnn_13_while_cond_4813968
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_2:
6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481396___redundant_placeholder0P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481396___redundant_placeholder1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481396___redundant_placeholder2P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_481396___redundant_placeholder3 
simple_rnn_13_while_identity
¶
simple_rnn_13/while/LessLesssimple_rnn_13_while_placeholder6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_13/while/Less
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_13/while/Identity"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
¸D

I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_481798
inputs_05
1simple_rnn_cell_13_matmul_readvariableop_resource6
2simple_rnn_cell_13_biasadd_readvariableop_resource7
3simple_rnn_cell_13_matmul_1_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2Æ
(simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype02*
(simple_rnn_cell_13/MatMul/ReadVariableOp¾
simple_rnn_cell_13/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMulÅ
)simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)simple_rnn_cell_13/BiasAdd/ReadVariableOpÍ
simple_rnn_cell_13/BiasAddBiasAdd#simple_rnn_cell_13/MatMul:product:01simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/BiasAddÌ
*simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*simple_rnn_cell_13/MatMul_1/ReadVariableOpº
simple_rnn_cell_13/MatMul_1MatMulzeros:output:02simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMul_1·
simple_rnn_cell_13/addAddV2#simple_rnn_cell_13/BiasAdd:output:0%simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/add
simple_rnn_cell_13/TanhTanhsimple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_13_matmul_readvariableop_resource2simple_rnn_cell_13_biasadd_readvariableop_resource3simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_481732*
condR
while_cond_481731*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1x
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}:::2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}
"
_user_specified_name
inputs/0
µ
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_480229

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@
 
_user_specified_nameinputs
Ä

.__inference_simple_rnn_13_layer_call_fn_481921
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_4796802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}
"
_user_specified_name
inputs/0
è

´
simple_rnn_13_while_cond_4808888
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_2:
6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480888___redundant_placeholder0P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480888___redundant_placeholder1P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480888___redundant_placeholder2P
Lsimple_rnn_13_while_simple_rnn_13_while_cond_480888___redundant_placeholder3 
simple_rnn_13_while_identity
¶
simple_rnn_13/while/LessLesssimple_rnn_13_while_placeholder6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_13/while/Less
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_13/while/Identity"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
©)
§
I__inference_sequential_19_layer_call_and_return_conditional_losses_480396

inputs
mel_stft_480363
mel_stft_480365
mel_stft_480367
simple_rnn_13_480372
simple_rnn_13_480374
simple_rnn_13_480376
dense_37_480380
dense_37_480382
dense_38_480385
dense_38_480387
dense_39_480390
dense_39_480392
identity¢ dense_37/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ mel_stft/StatefulPartitionedCall¢%simple_rnn_13/StatefulPartitionedCallâ
reshape_13/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_13_layer_call_and_return_conditional_losses_4798232
reshape_13/PartitionedCallÏ
 mel_stft/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0mel_stft_480363mel_stft_480365mel_stft_480367*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mel_stft_layer_call_and_return_conditional_losses_4799022"
 mel_stft/StatefulPartitionedCall
!normalization2d_7/PartitionedCallPartitionedCall)mel_stft/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_normalization2d_7_layer_call_and_return_conditional_losses_4799392#
!normalization2d_7/PartitionedCall
 squeeze_last_dim/PartitionedCallPartitionedCall*normalization2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_4799522"
 squeeze_last_dim/PartitionedCallï
%simple_rnn_13/StatefulPartitionedCallStatefulPartitionedCall)squeeze_last_dim/PartitionedCall:output:0simple_rnn_13_480372simple_rnn_13_480374simple_rnn_13_480376*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_4800812'
%simple_rnn_13/StatefulPartitionedCall
flatten_8/PartitionedCallPartitionedCall.simple_rnn_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4802292
flatten_8/PartitionedCall³
 dense_37/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_37_480380dense_37_480382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4802482"
 dense_37/StatefulPartitionedCallº
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_480385dense_38_480387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_4802752"
 dense_38/StatefulPartitionedCallº
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_480390dense_39_480392*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_4803022"
 dense_39/StatefulPartitionedCall±
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^mel_stft/StatefulPartitionedCall&^simple_rnn_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 mel_stft/StatefulPartitionedCall mel_stft/StatefulPartitionedCall2N
%simple_rnn_13/StatefulPartitionedCall%simple_rnn_13/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
ù
h
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_481676

inputs
identity{
SqueezeSqueezeinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP}:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
ç*
ï
while_body_480127
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_13_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_13_matmul_readvariableop_resource<
8while_simple_rnn_cell_13_biasadd_readvariableop_resource=
9while_simple_rnn_cell_13_matmul_1_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
.while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype020
.while/simple_rnn_cell_13/MatMul/ReadVariableOpè
while/simple_rnn_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/simple_rnn_cell_13/MatMulÙ
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpå
 while/simple_rnn_cell_13/BiasAddBiasAdd)while/simple_rnn_cell_13/MatMul:product:07while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 while/simple_rnn_cell_13/BiasAddà
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype022
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpÑ
!while/simple_rnn_cell_13/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!while/simple_rnn_cell_13/MatMul_1Ï
while/simple_rnn_cell_13/addAddV2)while/simple_rnn_cell_13/BiasAdd:output:0+while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/add
while/simple_rnn_cell_13/TanhTanh while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/Tanhå
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity!while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_13_biasadd_readvariableop_resource:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_13_matmul_1_readvariableop_resource;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_13_matmul_readvariableop_resource9while_simple_rnn_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
¹

!__inference__wrapped_model_479294
reshape_13_input>
:sequential_19_mel_stft_convolution_readvariableop_resource@
<sequential_19_mel_stft_convolution_1_readvariableop_resource:
6sequential_19_mel_stft_shape_1_readvariableop_resourceQ
Msequential_19_simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resourceR
Nsequential_19_simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceS
Osequential_19_simple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource9
5sequential_19_dense_37_matmul_readvariableop_resource:
6sequential_19_dense_37_biasadd_readvariableop_resource9
5sequential_19_dense_38_matmul_readvariableop_resource:
6sequential_19_dense_38_biasadd_readvariableop_resource9
5sequential_19_dense_39_matmul_readvariableop_resource:
6sequential_19_dense_39_biasadd_readvariableop_resource
identity¢!sequential_19/simple_rnn_13/while
sequential_19/reshape_13/ShapeShapereshape_13_input*
T0*
_output_shapes
:2 
sequential_19/reshape_13/Shape¦
,sequential_19/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_19/reshape_13/strided_slice/stackª
.sequential_19/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_19/reshape_13/strided_slice/stack_1ª
.sequential_19/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_19/reshape_13/strided_slice/stack_2ø
&sequential_19/reshape_13/strided_sliceStridedSlice'sequential_19/reshape_13/Shape:output:05sequential_19/reshape_13/strided_slice/stack:output:07sequential_19/reshape_13/strided_slice/stack_1:output:07sequential_19/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_19/reshape_13/strided_slice
(sequential_19/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_19/reshape_13/Reshape/shape/1
(sequential_19/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential_19/reshape_13/Reshape/shape/2
&sequential_19/reshape_13/Reshape/shapePack/sequential_19/reshape_13/strided_slice:output:01sequential_19/reshape_13/Reshape/shape/1:output:01sequential_19/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_19/reshape_13/Reshape/shapeÉ
 sequential_19/reshape_13/ReshapeReshapereshape_13_input/sequential_19/reshape_13/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2"
 sequential_19/reshape_13/Reshape­
*sequential_19/mel_stft/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2,
*sequential_19/mel_stft/strided_slice/stack±
,sequential_19/mel_stft/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2.
,sequential_19/mel_stft/strided_slice/stack_1±
,sequential_19/mel_stft/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential_19/mel_stft/strided_slice/stack_2
$sequential_19/mel_stft/strided_sliceStridedSlice)sequential_19/reshape_13/Reshape:output:03sequential_19/mel_stft/strided_slice/stack:output:05sequential_19/mel_stft/strided_slice/stack_1:output:05sequential_19/mel_stft/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask2&
$sequential_19/mel_stft/strided_slice£
%sequential_19/mel_stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_19/mel_stft/transpose/permç
 sequential_19/mel_stft/transpose	Transpose-sequential_19/mel_stft/strided_slice:output:0.sequential_19/mel_stft/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2"
 sequential_19/mel_stft/transpose
%sequential_19/mel_stft/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_19/mel_stft/ExpandDims/dimå
!sequential_19/mel_stft/ExpandDims
ExpandDims$sequential_19/mel_stft/transpose:y:0.sequential_19/mel_stft/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2#
!sequential_19/mel_stft/ExpandDimsë
1sequential_19/mel_stft/convolution/ReadVariableOpReadVariableOp:sequential_19_mel_stft_convolution_readvariableop_resource*(
_output_shapes
:*
dtype023
1sequential_19/mel_stft/convolution/ReadVariableOp
"sequential_19/mel_stft/convolutionConv2D*sequential_19/mel_stft/ExpandDims:output:09sequential_19/mel_stft/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2$
"sequential_19/mel_stft/convolutionñ
3sequential_19/mel_stft/convolution_1/ReadVariableOpReadVariableOp<sequential_19_mel_stft_convolution_1_readvariableop_resource*(
_output_shapes
:*
dtype025
3sequential_19/mel_stft/convolution_1/ReadVariableOp£
$sequential_19/mel_stft/convolution_1Conv2D*sequential_19/mel_stft/ExpandDims:output:0;sequential_19/mel_stft/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2&
$sequential_19/mel_stft/convolution_1
sequential_19/mel_stft/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sequential_19/mel_stft/pow/yÎ
sequential_19/mel_stft/powPow+sequential_19/mel_stft/convolution:output:0%sequential_19/mel_stft/pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
sequential_19/mel_stft/pow
sequential_19/mel_stft/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
sequential_19/mel_stft/pow_1/yÖ
sequential_19/mel_stft/pow_1Pow-sequential_19/mel_stft/convolution_1:output:0'sequential_19/mel_stft/pow_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
sequential_19/mel_stft/pow_1¾
sequential_19/mel_stft/addAddV2sequential_19/mel_stft/pow:z:0 sequential_19/mel_stft/pow_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
sequential_19/mel_stft/add«
'sequential_19/mel_stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'sequential_19/mel_stft/transpose_1/permâ
"sequential_19/mel_stft/transpose_1	Transposesequential_19/mel_stft/add:z:00sequential_19/mel_stft/transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2$
"sequential_19/mel_stft/transpose_1«
'sequential_19/mel_stft/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'sequential_19/mel_stft/transpose_2/permê
"sequential_19/mel_stft/transpose_2	Transpose&sequential_19/mel_stft/transpose_1:y:00sequential_19/mel_stft/transpose_2/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2$
"sequential_19/mel_stft/transpose_2
sequential_19/mel_stft/ShapeShape&sequential_19/mel_stft/transpose_2:y:0*
T0*
_output_shapes
:2
sequential_19/mel_stft/Shape£
sequential_19/mel_stft/unstackUnpack%sequential_19/mel_stft/Shape:output:0*
T0*
_output_shapes

: : : : *	
num2 
sequential_19/mel_stft/unstackÖ
-sequential_19/mel_stft/Shape_1/ReadVariableOpReadVariableOp6sequential_19_mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02/
-sequential_19/mel_stft/Shape_1/ReadVariableOp
sequential_19/mel_stft/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"  P   2 
sequential_19/mel_stft/Shape_1¥
 sequential_19/mel_stft/unstack_1Unpack'sequential_19/mel_stft/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2"
 sequential_19/mel_stft/unstack_1
$sequential_19/mel_stft/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2&
$sequential_19/mel_stft/Reshape/shapeÕ
sequential_19/mel_stft/ReshapeReshape&sequential_19/mel_stft/transpose_2:y:0-sequential_19/mel_stft/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_19/mel_stft/ReshapeÞ
1sequential_19/mel_stft/transpose_3/ReadVariableOpReadVariableOp6sequential_19_mel_stft_shape_1_readvariableop_resource*
_output_shapes
:	P*
dtype023
1sequential_19/mel_stft/transpose_3/ReadVariableOp£
'sequential_19/mel_stft/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'sequential_19/mel_stft/transpose_3/permì
"sequential_19/mel_stft/transpose_3	Transpose9sequential_19/mel_stft/transpose_3/ReadVariableOp:value:00sequential_19/mel_stft/transpose_3/perm:output:0*
T0*
_output_shapes
:	P2$
"sequential_19/mel_stft/transpose_3¡
&sequential_19/mel_stft/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ÿÿÿÿ2(
&sequential_19/mel_stft/Reshape_1/shapeÒ
 sequential_19/mel_stft/Reshape_1Reshape&sequential_19/mel_stft/transpose_3:y:0/sequential_19/mel_stft/Reshape_1/shape:output:0*
T0*
_output_shapes
:	P2"
 sequential_19/mel_stft/Reshape_1Î
sequential_19/mel_stft/MatMulMatMul'sequential_19/mel_stft/Reshape:output:0)sequential_19/mel_stft/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP2
sequential_19/mel_stft/MatMul
(sequential_19/mel_stft/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_19/mel_stft/Reshape_2/shape/1
(sequential_19/mel_stft/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}2*
(sequential_19/mel_stft/Reshape_2/shape/2
(sequential_19/mel_stft/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :P2*
(sequential_19/mel_stft/Reshape_2/shape/3È
&sequential_19/mel_stft/Reshape_2/shapePack'sequential_19/mel_stft/unstack:output:01sequential_19/mel_stft/Reshape_2/shape/1:output:01sequential_19/mel_stft/Reshape_2/shape/2:output:01sequential_19/mel_stft/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_19/mel_stft/Reshape_2/shapeã
 sequential_19/mel_stft/Reshape_2Reshape'sequential_19/mel_stft/MatMul:product:0/sequential_19/mel_stft/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P2"
 sequential_19/mel_stft/Reshape_2«
'sequential_19/mel_stft/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'sequential_19/mel_stft/transpose_4/permì
"sequential_19/mel_stft/transpose_4	Transpose)sequential_19/mel_stft/Reshape_2:output:00sequential_19/mel_stft/transpose_4/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2$
"sequential_19/mel_stft/transpose_4
sequential_19/mel_stft/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_19/mel_stft/Const
sequential_19/mel_stft/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2 
sequential_19/mel_stft/Const_1ò
,sequential_19/mel_stft/clip_by_value/MinimumMinimum&sequential_19/mel_stft/transpose_4:y:0'sequential_19/mel_stft/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2.
,sequential_19/mel_stft/clip_by_value/Minimumê
$sequential_19/mel_stft/clip_by_valueMaximum0sequential_19/mel_stft/clip_by_value/Minimum:z:0%sequential_19/mel_stft/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2&
$sequential_19/mel_stft/clip_by_value¦
sequential_19/mel_stft/SqrtSqrt(sequential_19/mel_stft/clip_by_value:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
sequential_19/mel_stft/Sqrt
sequential_19/mel_stft/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
sequential_19/mel_stft/Pow_2/yÇ
sequential_19/mel_stft/Pow_2Powsequential_19/mel_stft/Sqrt:y:0'sequential_19/mel_stft/Pow_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
sequential_19/mel_stft/Pow_2
 sequential_19/mel_stft/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2"
 sequential_19/mel_stft/Maximum/yÒ
sequential_19/mel_stft/MaximumMaximum sequential_19/mel_stft/Pow_2:z:0)sequential_19/mel_stft/Maximum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
sequential_19/mel_stft/Maximum
sequential_19/mel_stft/LogLog"sequential_19/mel_stft/Maximum:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
sequential_19/mel_stft/Log
sequential_19/mel_stft/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
sequential_19/mel_stft/mul/xÀ
sequential_19/mel_stft/mulMul%sequential_19/mel_stft/mul/x:output:0sequential_19/mel_stft/Log:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
sequential_19/mel_stft/mul
 sequential_19/mel_stft/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *]@2"
 sequential_19/mel_stft/truediv/yÐ
sequential_19/mel_stft/truedivRealDivsequential_19/mel_stft/mul:z:0)sequential_19/mel_stft/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2 
sequential_19/mel_stft/truediv±
,sequential_19/mel_stft/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential_19/mel_stft/Max/reduction_indiceså
sequential_19/mel_stft/MaxMax"sequential_19/mel_stft/truediv:z:05sequential_19/mel_stft/Max/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
sequential_19/mel_stft/MaxÂ
sequential_19/mel_stft/subSub"sequential_19/mel_stft/truediv:z:0#sequential_19/mel_stft/Max:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
sequential_19/mel_stft/sub
"sequential_19/mel_stft/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â2$
"sequential_19/mel_stft/Maximum_1/yÖ
 sequential_19/mel_stft/Maximum_1Maximumsequential_19/mel_stft/sub:z:0+sequential_19/mel_stft/Maximum_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2"
 sequential_19/mel_stft/Maximum_1Å
6sequential_19/normalization2d_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         28
6sequential_19/normalization2d_7/Mean/reduction_indices
$sequential_19/normalization2d_7/MeanMean$sequential_19/mel_stft/Maximum_1:z:0?sequential_19/normalization2d_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2&
$sequential_19/normalization2d_7/Meanû
Qsequential_19/normalization2d_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2S
Qsequential_19/normalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices×
?sequential_19/normalization2d_7/reduce_std/reduce_variance/MeanMean$sequential_19/mel_stft/Maximum_1:z:0Zsequential_19/normalization2d_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2A
?sequential_19/normalization2d_7/reduce_std/reduce_variance/Mean±
>sequential_19/normalization2d_7/reduce_std/reduce_variance/subSub$sequential_19/mel_stft/Maximum_1:z:0Hsequential_19/normalization2d_7/reduce_std/reduce_variance/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2@
>sequential_19/normalization2d_7/reduce_std/reduce_variance/sub
Asequential_19/normalization2d_7/reduce_std/reduce_variance/SquareSquareBsequential_19/normalization2d_7/reduce_std/reduce_variance/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2C
Asequential_19/normalization2d_7/reduce_std/reduce_variance/Squareÿ
Ssequential_19/normalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2U
Ssequential_19/normalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indicesþ
Asequential_19/normalization2d_7/reduce_std/reduce_variance/Mean_1MeanEsequential_19/normalization2d_7/reduce_std/reduce_variance/Square:y:0\sequential_19/normalization2d_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2C
Asequential_19/normalization2d_7/reduce_std/reduce_variance/Mean_1ð
/sequential_19/normalization2d_7/reduce_std/SqrtSqrtJsequential_19/normalization2d_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_19/normalization2d_7/reduce_std/Sqrtà
#sequential_19/normalization2d_7/subSub$sequential_19/mel_stft/Maximum_1:z:0-sequential_19/normalization2d_7/Mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2%
#sequential_19/normalization2d_7/sub
%sequential_19/normalization2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2'
%sequential_19/normalization2d_7/add/yò
#sequential_19/normalization2d_7/addAddV23sequential_19/normalization2d_7/reduce_std/Sqrt:y:0.sequential_19/normalization2d_7/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_19/normalization2d_7/addé
'sequential_19/normalization2d_7/truedivRealDiv'sequential_19/normalization2d_7/sub:z:0'sequential_19/normalization2d_7/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2)
'sequential_19/normalization2d_7/truedivÞ
&sequential_19/squeeze_last_dim/SqueezeSqueeze+sequential_19/normalization2d_7/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2(
&sequential_19/squeeze_last_dim/Squeeze¥
!sequential_19/simple_rnn_13/ShapeShape/sequential_19/squeeze_last_dim/Squeeze:output:0*
T0*
_output_shapes
:2#
!sequential_19/simple_rnn_13/Shape¬
/sequential_19/simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_19/simple_rnn_13/strided_slice/stack°
1sequential_19/simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_19/simple_rnn_13/strided_slice/stack_1°
1sequential_19/simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_19/simple_rnn_13/strided_slice/stack_2
)sequential_19/simple_rnn_13/strided_sliceStridedSlice*sequential_19/simple_rnn_13/Shape:output:08sequential_19/simple_rnn_13/strided_slice/stack:output:0:sequential_19/simple_rnn_13/strided_slice/stack_1:output:0:sequential_19/simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_19/simple_rnn_13/strided_slice
'sequential_19/simple_rnn_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_19/simple_rnn_13/zeros/mul/yÜ
%sequential_19/simple_rnn_13/zeros/mulMul2sequential_19/simple_rnn_13/strided_slice:output:00sequential_19/simple_rnn_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_19/simple_rnn_13/zeros/mul
(sequential_19/simple_rnn_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2*
(sequential_19/simple_rnn_13/zeros/Less/y×
&sequential_19/simple_rnn_13/zeros/LessLess)sequential_19/simple_rnn_13/zeros/mul:z:01sequential_19/simple_rnn_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2(
&sequential_19/simple_rnn_13/zeros/Less
*sequential_19/simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2,
*sequential_19/simple_rnn_13/zeros/packed/1ó
(sequential_19/simple_rnn_13/zeros/packedPack2sequential_19/simple_rnn_13/strided_slice:output:03sequential_19/simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(sequential_19/simple_rnn_13/zeros/packed
'sequential_19/simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'sequential_19/simple_rnn_13/zeros/Constå
!sequential_19/simple_rnn_13/zerosFill1sequential_19/simple_rnn_13/zeros/packed:output:00sequential_19/simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential_19/simple_rnn_13/zeros­
*sequential_19/simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_19/simple_rnn_13/transpose/perm÷
%sequential_19/simple_rnn_13/transpose	Transpose/sequential_19/squeeze_last_dim/Squeeze:output:03sequential_19/simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2'
%sequential_19/simple_rnn_13/transpose£
#sequential_19/simple_rnn_13/Shape_1Shape)sequential_19/simple_rnn_13/transpose:y:0*
T0*
_output_shapes
:2%
#sequential_19/simple_rnn_13/Shape_1°
1sequential_19/simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_19/simple_rnn_13/strided_slice_1/stack´
3sequential_19/simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_19/simple_rnn_13/strided_slice_1/stack_1´
3sequential_19/simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_19/simple_rnn_13/strided_slice_1/stack_2
+sequential_19/simple_rnn_13/strided_slice_1StridedSlice,sequential_19/simple_rnn_13/Shape_1:output:0:sequential_19/simple_rnn_13/strided_slice_1/stack:output:0<sequential_19/simple_rnn_13/strided_slice_1/stack_1:output:0<sequential_19/simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_19/simple_rnn_13/strided_slice_1½
7sequential_19/simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ29
7sequential_19/simple_rnn_13/TensorArrayV2/element_shape¢
)sequential_19/simple_rnn_13/TensorArrayV2TensorListReserve@sequential_19/simple_rnn_13/TensorArrayV2/element_shape:output:04sequential_19/simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_19/simple_rnn_13/TensorArrayV2÷
Qsequential_19/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2S
Qsequential_19/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeè
Csequential_19/simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_19/simple_rnn_13/transpose:y:0Zsequential_19/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02E
Csequential_19/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor°
1sequential_19/simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_19/simple_rnn_13/strided_slice_2/stack´
3sequential_19/simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_19/simple_rnn_13/strided_slice_2/stack_1´
3sequential_19/simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_19/simple_rnn_13/strided_slice_2/stack_2¤
+sequential_19/simple_rnn_13/strided_slice_2StridedSlice)sequential_19/simple_rnn_13/transpose:y:0:sequential_19/simple_rnn_13/strided_slice_2/stack:output:0<sequential_19/simple_rnn_13/strided_slice_2/stack_1:output:0<sequential_19/simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2-
+sequential_19/simple_rnn_13/strided_slice_2
Dsequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOpMsequential_19_simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype02F
Dsequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOp®
5sequential_19/simple_rnn_13/simple_rnn_cell_13/MatMulMatMul4sequential_19/simple_rnn_13/strided_slice_2:output:0Lsequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5sequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul
Esequential_19/simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOpNsequential_19_simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_19/simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp½
6sequential_19/simple_rnn_13/simple_rnn_cell_13/BiasAddBiasAdd?sequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul:product:0Msequential_19/simple_rnn_13/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@28
6sequential_19/simple_rnn_13/simple_rnn_cell_13/BiasAdd 
Fsequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpOsequential_19_simple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02H
Fsequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOpª
7sequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul_1MatMul*sequential_19/simple_rnn_13/zeros:output:0Nsequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@29
7sequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul_1§
2sequential_19/simple_rnn_13/simple_rnn_cell_13/addAddV2?sequential_19/simple_rnn_13/simple_rnn_cell_13/BiasAdd:output:0Asequential_19/simple_rnn_13/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2sequential_19/simple_rnn_13/simple_rnn_cell_13/addÜ
3sequential_19/simple_rnn_13/simple_rnn_cell_13/TanhTanh6sequential_19/simple_rnn_13/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@25
3sequential_19/simple_rnn_13/simple_rnn_cell_13/TanhÇ
9sequential_19/simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2;
9sequential_19/simple_rnn_13/TensorArrayV2_1/element_shape¨
+sequential_19/simple_rnn_13/TensorArrayV2_1TensorListReserveBsequential_19/simple_rnn_13/TensorArrayV2_1/element_shape:output:04sequential_19/simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+sequential_19/simple_rnn_13/TensorArrayV2_1
 sequential_19/simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_19/simple_rnn_13/time·
4sequential_19/simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential_19/simple_rnn_13/while/maximum_iterations¢
.sequential_19/simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_19/simple_rnn_13/while/loop_counterÏ
!sequential_19/simple_rnn_13/whileWhile7sequential_19/simple_rnn_13/while/loop_counter:output:0=sequential_19/simple_rnn_13/while/maximum_iterations:output:0)sequential_19/simple_rnn_13/time:output:04sequential_19/simple_rnn_13/TensorArrayV2_1:handle:0*sequential_19/simple_rnn_13/zeros:output:04sequential_19/simple_rnn_13/strided_slice_1:output:0Ssequential_19/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_19_simple_rnn_13_simple_rnn_cell_13_matmul_readvariableop_resourceNsequential_19_simple_rnn_13_simple_rnn_cell_13_biasadd_readvariableop_resourceOsequential_19_simple_rnn_13_simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*9
body1R/
-sequential_19_simple_rnn_13_while_body_479205*9
cond1R/
-sequential_19_simple_rnn_13_while_cond_479204*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2#
!sequential_19/simple_rnn_13/whileí
Lsequential_19/simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2N
Lsequential_19/simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeØ
>sequential_19/simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_19/simple_rnn_13/while:output:3Usequential_19/simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype02@
>sequential_19/simple_rnn_13/TensorArrayV2Stack/TensorListStack¹
1sequential_19/simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ23
1sequential_19/simple_rnn_13/strided_slice_3/stack´
3sequential_19/simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_19/simple_rnn_13/strided_slice_3/stack_1´
3sequential_19/simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_19/simple_rnn_13/strided_slice_3/stack_2Â
+sequential_19/simple_rnn_13/strided_slice_3StridedSliceGsequential_19/simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_19/simple_rnn_13/strided_slice_3/stack:output:0<sequential_19/simple_rnn_13/strided_slice_3/stack_1:output:0<sequential_19/simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2-
+sequential_19/simple_rnn_13/strided_slice_3±
,sequential_19/simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,sequential_19/simple_rnn_13/transpose_1/perm
'sequential_19/simple_rnn_13/transpose_1	TransposeGsequential_19/simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:05sequential_19/simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2)
'sequential_19/simple_rnn_13/transpose_1
sequential_19/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential_19/flatten_8/ConstÕ
sequential_19/flatten_8/ReshapeReshape+sequential_19/simple_rnn_13/transpose_1:y:0&sequential_19/flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2!
sequential_19/flatten_8/ReshapeÓ
,sequential_19/dense_37/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_37_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype02.
,sequential_19/dense_37/MatMul/ReadVariableOpÚ
sequential_19/dense_37/MatMulMatMul(sequential_19/flatten_8/Reshape:output:04sequential_19/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_19/dense_37/MatMulÑ
-sequential_19/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_19/dense_37/BiasAdd/ReadVariableOpÝ
sequential_19/dense_37/BiasAddBiasAdd'sequential_19/dense_37/MatMul:product:05sequential_19/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential_19/dense_37/BiasAdd
sequential_19/dense_37/ReluRelu'sequential_19/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_19/dense_37/ReluÒ
,sequential_19/dense_38/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02.
,sequential_19/dense_38/MatMul/ReadVariableOpÛ
sequential_19/dense_38/MatMulMatMul)sequential_19/dense_37/Relu:activations:04sequential_19/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_19/dense_38/MatMulÑ
-sequential_19/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_19/dense_38/BiasAdd/ReadVariableOpÝ
sequential_19/dense_38/BiasAddBiasAdd'sequential_19/dense_38/MatMul:product:05sequential_19/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_19/dense_38/BiasAdd
sequential_19/dense_38/ReluRelu'sequential_19/dense_38/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_19/dense_38/ReluÒ
,sequential_19/dense_39/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_39_matmul_readvariableop_resource*
_output_shapes

: $*
dtype02.
,sequential_19/dense_39/MatMul/ReadVariableOpÛ
sequential_19/dense_39/MatMulMatMul)sequential_19/dense_38/Relu:activations:04sequential_19/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
sequential_19/dense_39/MatMulÑ
-sequential_19/dense_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_39_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02/
-sequential_19/dense_39/BiasAdd/ReadVariableOpÝ
sequential_19/dense_39/BiasAddBiasAdd'sequential_19/dense_39/MatMul:product:05sequential_19/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2 
sequential_19/dense_39/BiasAdd¦
sequential_19/dense_39/SoftmaxSoftmax'sequential_19/dense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2 
sequential_19/dense_39/Softmax 
IdentityIdentity(sequential_19/dense_39/Softmax:softmax:0"^sequential_19/simple_rnn_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::2F
!sequential_19/simple_rnn_13/while!sequential_19/simple_rnn_13/while:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
*
_user_specified_namereshape_13_input
	

$__inference_signature_wrapper_480528
reshape_13_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallreshape_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_4792942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ}::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
*
_user_specified_namereshape_13_input
>
×
D__inference_mel_stft_layer_call_and_return_conditional_losses_481633
x'
#convolution_readvariableop_resource)
%convolution_1_readvariableop_resource#
shape_1_readvariableop_resource
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2õ
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask2
strided_sliceu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposestrided_slice:output:0transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimstranspose:y:0ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2

ExpandDims¦
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*(
_output_shapes
:*
dtype02
convolution/ReadVariableOpÁ
convolutionConv2DExpandDims:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
convolution¬
convolution_1/ReadVariableOpReadVariableOp%convolution_1_readvariableop_resource*(
_output_shapes
:*
dtype02
convolution_1/ReadVariableOpÇ
convolution_1Conv2DExpandDims:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
paddingSAME*
strides	
2
convolution_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yr
powPowconvolution:output:0pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
powW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yz
pow_1Powconvolution_1:output:0pow_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
pow_1b
addAddV2pow:z:0	pow_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
add}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/perm
transpose_1	Transposeadd:z:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
transpose_1}
transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_2/perm
transpose_2	Transposetranspose_1:y:0transpose_2/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2
transpose_2M
ShapeShapetranspose_2:y:0*
T0*
_output_shapes
:2
Shape^
unstackUnpackShape:output:0*
T0*
_output_shapes

: : : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"  P   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Reshape/shapey
ReshapeReshapetranspose_2:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshape
transpose_3/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	P*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes
:	P2
transpose_3s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ÿÿÿÿ2
Reshape_1/shapev
	Reshape_1Reshapetranspose_3:y:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	P2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}2
Reshape_2/shape/2h
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :P2
Reshape_2/shape/3¾
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P2
	Reshape_2}
transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_4/perm
transpose_4	TransposeReshape_2:output:0transpose_4/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
transpose_4S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1
clip_by_value/MinimumMinimumtranspose_4:y:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
clip_by_value/Minimum
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
clip_by_valuea
SqrtSqrtclip_by_value:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
SqrtW
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
Pow_2/yk
Pow_2PowSqrt:y:0Pow_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
Pow_2[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
	Maximum/yv
MaximumMaximum	Pow_2:z:0Maximum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2	
MaximumX
LogLogMaximum:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
LogS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/xd
mulMulmul/x:output:0Log:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *]@2
	truediv/yt
truedivRealDivmul:z:0truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2	
truediv
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Max/reduction_indices
MaxMaxtruediv:z:0Max/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Maxf
subSubtruediv:z:0Max:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
sub_
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â2
Maximum_1/yz
	Maximum_1Maximumsub:z:0Maximum_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2
	Maximum_1i
IdentityIdentityMaximum_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}::::O K
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}

_user_specified_namex
ùC

I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_482044

inputs5
1simple_rnn_cell_13_matmul_readvariableop_resource6
2simple_rnn_cell_13_biasadd_readvariableop_resource7
3simple_rnn_cell_13_matmul_1_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ}2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
shrink_axis_mask2
strided_slice_2Æ
(simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_13_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype02*
(simple_rnn_cell_13/MatMul/ReadVariableOp¾
simple_rnn_cell_13/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMulÅ
)simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)simple_rnn_cell_13/BiasAdd/ReadVariableOpÍ
simple_rnn_cell_13/BiasAddBiasAdd#simple_rnn_cell_13/MatMul:product:01simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/BiasAddÌ
*simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_13_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*simple_rnn_cell_13/MatMul_1/ReadVariableOpº
simple_rnn_cell_13/MatMul_1MatMulzeros:output:02simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/MatMul_1·
simple_rnn_cell_13/addAddV2#simple_rnn_cell_13/BiasAdd:output:0%simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/add
simple_rnn_cell_13/TanhTanhsimple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
simple_rnn_cell_13/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_13_matmul_readvariableop_resource2simple_rnn_cell_13_biasadd_readvariableop_resource3simple_rnn_cell_13_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_481978*
condR
while_cond_481977*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:Pÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2
transpose_1o
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿP}:::2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
 
_user_specified_nameinputs
Ð
ª
while_cond_480126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_480126___redundant_placeholder04
0while_while_cond_480126___redundant_placeholder14
0while_while_cond_480126___redundant_placeholder24
0while_while_cond_480126___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
8
é	
simple_rnn_13_while_body_4811728
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_27
3simple_rnn_13_while_simple_rnn_13_strided_slice_1_0s
osimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0L
Hsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0M
Isimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0 
simple_rnn_13_while_identity"
simple_rnn_13_while_identity_1"
simple_rnn_13_while_identity_2"
simple_rnn_13_while_identity_3"
simple_rnn_13_while_identity_45
1simple_rnn_13_while_simple_rnn_13_strided_slice_1q
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceJ
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceK
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceß
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   2G
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape§
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_13_while_placeholderNsimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype029
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype02>
<simple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp 
-simple_rnn_13/while/simple_rnn_cell_13/MatMulMatMul>simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_13/while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2/
-simple_rnn_13/while/simple_rnn_cell_13/MatMul
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02?
=simple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp
.simple_rnn_13/while/simple_rnn_cell_13/BiasAddBiasAdd7simple_rnn_13/while/simple_rnn_cell_13/MatMul:product:0Esimple_rnn_13/while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.simple_rnn_13/while/simple_rnn_cell_13/BiasAdd
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02@
>simple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1MatMul!simple_rnn_13_while_placeholder_2Fsimple_rnn_13/while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@21
/simple_rnn_13/while/simple_rnn_cell_13/MatMul_1
*simple_rnn_13/while/simple_rnn_cell_13/addAddV27simple_rnn_13/while/simple_rnn_cell_13/BiasAdd:output:09simple_rnn_13/while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*simple_rnn_13/while/simple_rnn_cell_13/addÄ
+simple_rnn_13/while/simple_rnn_cell_13/TanhTanh.simple_rnn_13/while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+simple_rnn_13/while/simple_rnn_cell_13/Tanh«
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_13_while_placeholder_1simple_rnn_13_while_placeholder/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add/y¡
simple_rnn_13/while/addAddV2simple_rnn_13_while_placeholder"simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add|
simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_13/while/add_1/y¼
simple_rnn_13/while/add_1AddV24simple_rnn_13_while_simple_rnn_13_while_loop_counter$simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/add_1
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_13/while/Identity©
simple_rnn_13/while/Identity_1Identity:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_1
simple_rnn_13/while/Identity_2Identitysimple_rnn_13/while/add:z:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_2·
simple_rnn_13/while/Identity_3IdentityHsimple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2 
simple_rnn_13/while/Identity_3¯
simple_rnn_13/while/Identity_4Identity/simple_rnn_13/while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
simple_rnn_13/while/Identity_4"E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0"I
simple_rnn_13_while_identity_1'simple_rnn_13/while/Identity_1:output:0"I
simple_rnn_13_while_identity_2'simple_rnn_13/while/Identity_2:output:0"I
simple_rnn_13_while_identity_3'simple_rnn_13/while/Identity_3:output:0"I
simple_rnn_13_while_identity_4'simple_rnn_13/while/Identity_4:output:0"h
1simple_rnn_13_while_simple_rnn_13_strided_slice_13simple_rnn_13_while_simple_rnn_13_strided_slice_1_0"
Fsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resourceHsimple_rnn_13_while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"
Gsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resourceIsimple_rnn_13_while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"
Esimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resourceGsimple_rnn_13_while_simple_rnn_cell_13_matmul_readvariableop_resource_0"à
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ð
ª
while_cond_481977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_481977___redundant_placeholder04
0while_while_cond_481977___redundant_placeholder14
0while_while_cond_481977___redundant_placeholder24
0while_while_cond_481977___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ð
ª
while_cond_482089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_482089___redundant_placeholder04
0while_while_cond_482089___redundant_placeholder14
0while_while_cond_482089___redundant_placeholder24
0while_while_cond_482089___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
µ#

while_body_479734
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_13_479756_0%
!while_simple_rnn_cell_13_479758_0%
!while_simple_rnn_cell_13_479760_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_13_479756#
while_simple_rnn_cell_13_479758#
while_simple_rnn_cell_13_479760¢0while/simple_rnn_cell_13/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÞ
0while/simple_rnn_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_13_479756_0!while_simple_rnn_cell_13_479758_0!while_simple_rnn_cell_13_479760_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_47936022
0while/simple_rnn_cell_13/StatefulPartitionedCallý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity¤
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2À
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ð
while/Identity_4Identity9while/simple_rnn_cell_13/StatefulPartitionedCall:output:11^while/simple_rnn_cell_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_13_479756!while_simple_rnn_cell_13_479756_0"D
while_simple_rnn_cell_13_479758!while_simple_rnn_cell_13_479758_0"D
while_simple_rnn_cell_13_479760!while_simple_rnn_cell_13_479760_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::2d
0while/simple_rnn_cell_13/StatefulPartitionedCall0while/simple_rnn_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ç*
ï
while_body_481978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_13_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_13_matmul_readvariableop_resource<
8while_simple_rnn_cell_13_biasadd_readvariableop_resource=
9while_simple_rnn_cell_13_matmul_1_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
.while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype020
.while/simple_rnn_cell_13/MatMul/ReadVariableOpè
while/simple_rnn_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/simple_rnn_cell_13/MatMulÙ
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpå
 while/simple_rnn_cell_13/BiasAddBiasAdd)while/simple_rnn_cell_13/MatMul:product:07while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 while/simple_rnn_cell_13/BiasAddà
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype022
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpÑ
!while/simple_rnn_cell_13/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!while/simple_rnn_cell_13/MatMul_1Ï
while/simple_rnn_cell_13/addAddV2)while/simple_rnn_cell_13/BiasAdd:output:0+while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/add
while/simple_rnn_cell_13/TanhTanh while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/Tanhå
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity!while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_13_biasadd_readvariableop_resource:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_13_matmul_1_readvariableop_resource;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_13_matmul_readvariableop_resource9while_simple_rnn_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
±
¬
D__inference_dense_39_layer_call_and_return_conditional_losses_482240

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: $*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì
b
F__inference_reshape_13_layer_call_and_return_conditional_losses_479823

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1m
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
øÁ
¬
"__inference__traced_restore_482614
file_prefix*
&assignvariableop_mel_stft_real_kernels,
(assignvariableop_1_mel_stft_imag_kernels(
$assignvariableop_2_mel_stft_variable&
"assignvariableop_3_dense_37_kernel$
 assignvariableop_4_dense_37_bias&
"assignvariableop_5_dense_38_kernel$
 assignvariableop_6_dense_38_bias&
"assignvariableop_7_dense_39_kernel$
 assignvariableop_8_dense_39_bias
assignvariableop_9_beta_1
assignvariableop_10_beta_2
assignvariableop_11_decay%
!assignvariableop_12_learning_rate!
assignvariableop_13_adam_iter?
;assignvariableop_14_simple_rnn_13_simple_rnn_cell_13_kernelI
Eassignvariableop_15_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel=
9assignvariableop_16_simple_rnn_13_simple_rnn_cell_13_bias
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_14
0assignvariableop_21_adam_mel_stft_real_kernels_m4
0assignvariableop_22_adam_mel_stft_imag_kernels_m0
,assignvariableop_23_adam_mel_stft_variable_m.
*assignvariableop_24_adam_dense_37_kernel_m,
(assignvariableop_25_adam_dense_37_bias_m.
*assignvariableop_26_adam_dense_38_kernel_m,
(assignvariableop_27_adam_dense_38_bias_m.
*assignvariableop_28_adam_dense_39_kernel_m,
(assignvariableop_29_adam_dense_39_bias_mF
Bassignvariableop_30_adam_simple_rnn_13_simple_rnn_cell_13_kernel_mP
Lassignvariableop_31_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_mD
@assignvariableop_32_adam_simple_rnn_13_simple_rnn_cell_13_bias_m4
0assignvariableop_33_adam_mel_stft_real_kernels_v4
0assignvariableop_34_adam_mel_stft_imag_kernels_v0
,assignvariableop_35_adam_mel_stft_variable_v.
*assignvariableop_36_adam_dense_37_kernel_v,
(assignvariableop_37_adam_dense_37_bias_v.
*assignvariableop_38_adam_dense_38_kernel_v,
(assignvariableop_39_adam_dense_38_bias_v.
*assignvariableop_40_adam_dense_39_kernel_v,
(assignvariableop_41_adam_dense_39_bias_vF
Bassignvariableop_42_adam_simple_rnn_13_simple_rnn_cell_13_kernel_vP
Lassignvariableop_43_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_vD
@assignvariableop_44_adam_simple_rnn_13_simple_rnn_cell_13_bias_v
identity_46¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Þ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*ê
valueàBÝ.B@layer_with_weights-0/dft_real_kernels/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/dft_imag_kernels/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/freq2mel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_real_kernels/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_imag_kernels/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/freq2mel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_real_kernels/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/dft_imag_kernels/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/freq2mel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesê
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¥
AssignVariableOpAssignVariableOp&assignvariableop_mel_stft_real_kernelsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1­
AssignVariableOp_1AssignVariableOp(assignvariableop_1_mel_stft_imag_kernelsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_mel_stft_variableIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_37_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_37_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_38_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¥
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_38_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_39_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¥
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_39_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¢
AssignVariableOp_10AssignVariableOpassignvariableop_10_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13¥
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ã
AssignVariableOp_14AssignVariableOp;assignvariableop_14_simple_rnn_13_simple_rnn_cell_13_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Í
AssignVariableOp_15AssignVariableOpEassignvariableop_15_simple_rnn_13_simple_rnn_cell_13_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Á
AssignVariableOp_16AssignVariableOp9assignvariableop_16_simple_rnn_13_simple_rnn_cell_13_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¸
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_mel_stft_real_kernels_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¸
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_mel_stft_imag_kernels_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23´
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_mel_stft_variable_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_37_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_37_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26²
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_38_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27°
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_38_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_39_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29°
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_39_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ê
AssignVariableOp_30AssignVariableOpBassignvariableop_30_adam_simple_rnn_13_simple_rnn_cell_13_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ô
AssignVariableOp_31AssignVariableOpLassignvariableop_31_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32È
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_simple_rnn_13_simple_rnn_cell_13_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¸
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_mel_stft_real_kernels_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¸
AssignVariableOp_34AssignVariableOp0assignvariableop_34_adam_mel_stft_imag_kernels_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35´
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_mel_stft_variable_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36²
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_37_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37°
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense_37_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38²
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_38_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39°
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_38_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40²
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_39_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41°
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_39_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ê
AssignVariableOp_42AssignVariableOpBassignvariableop_42_adam_simple_rnn_13_simple_rnn_cell_13_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ô
AssignVariableOp_43AssignVariableOpLassignvariableop_43_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44È
AssignVariableOp_44AssignVariableOp@assignvariableop_44_adam_simple_rnn_13_simple_rnn_cell_13_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¼
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45¯
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*Ë
_input_shapes¹
¶: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ç*
ï
while_body_481732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_13_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_13_matmul_readvariableop_resource<
8while_simple_rnn_cell_13_biasadd_readvariableop_resource=
9while_simple_rnn_cell_13_matmul_1_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ}   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
.while/simple_rnn_cell_13/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_13_matmul_readvariableop_resource_0*
_output_shapes

:}@*
dtype020
.while/simple_rnn_cell_13/MatMul/ReadVariableOpè
while/simple_rnn_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/simple_rnn_cell_13/MatMulÙ
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/while/simple_rnn_cell_13/BiasAdd/ReadVariableOpå
 while/simple_rnn_cell_13/BiasAddBiasAdd)while/simple_rnn_cell_13/MatMul:product:07while/simple_rnn_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 while/simple_rnn_cell_13/BiasAddà
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype022
0while/simple_rnn_cell_13/MatMul_1/ReadVariableOpÑ
!while/simple_rnn_cell_13/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!while/simple_rnn_cell_13/MatMul_1Ï
while/simple_rnn_cell_13/addAddV2)while/simple_rnn_cell_13/BiasAdd:output:0+while/simple_rnn_cell_13/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/add
while/simple_rnn_cell_13/TanhTanh while/simple_rnn_cell_13/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/simple_rnn_cell_13/Tanhå
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_13/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity!while/simple_rnn_cell_13/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_13_biasadd_readvariableop_resource:while_simple_rnn_cell_13_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_13_matmul_1_readvariableop_resource;while_simple_rnn_cell_13_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_13_matmul_readvariableop_resource9while_simple_rnn_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: "¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
N
reshape_13_input:
"serving_default_reshape_13_input:0ÿÿÿÿÿÿÿÿÿ}<
dense_390
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ$tensorflow/serving/predict:¹Ì
ÀE
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api

signatures
¸_default_save_signature
+¹&call_and_return_all_conditional_losses
º__call__"ñA
_tf_keras_sequentialÒA{"class_name": "Sequential", "name": "sequential_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_13_input"}}, {"class_name": "Reshape", "config": {"name": "reshape_13", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, -1]}}}, {"class_name": "Melspectrogram", "config": {"name": "mel_stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 16000]}, "dtype": "float32", "n_dft": 1024, "n_hop": 128, "padding": "same", "power_spectrogram": 2.0, "return_decibel_spectrogram": false, "trainable_kernel": false, "image_data_format": "channels_last", "sr": 16000, "n_mels": 80, "fmin": 40.0, "fmax": 8000.0, "trainable_fb": false, "power_melgram": 1.0, "return_decibel_melgram": true, "htk": false, "norm": "slaney"}}, {"class_name": "Normalization2D", "config": {"name": "normalization2d_7", "trainable": true, "dtype": "float32", "int_axis": 0, "str_axis": null, "image_data_format": "channels_last"}}, {"class_name": "Lambda", "config": {"name": "squeeze_last_dim", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABDAAAAcwwAAAB0AKABfABkAaECUwApAk7p/////ykC2gFL2gdzcXVl\nZXplKQHaAXGpAHIFAAAA+j9DOi9Vc2Vycy9zenltby9BcHBEYXRhL0xvY2FsL1RlbXAvaXB5a2Vy\nbmVsXzE5OTk2LzIxNjA0MjA1NjIucHnaCDxsYW1iZGE+GAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_13", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_13_input"}}, {"class_name": "Reshape", "config": {"name": "reshape_13", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, -1]}}}, {"class_name": "Melspectrogram", "config": {"name": "mel_stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 16000]}, "dtype": "float32", "n_dft": 1024, "n_hop": 128, "padding": "same", "power_spectrogram": 2.0, "return_decibel_spectrogram": false, "trainable_kernel": false, "image_data_format": "channels_last", "sr": 16000, "n_mels": 80, "fmin": 40.0, "fmax": 8000.0, "trainable_fb": false, "power_melgram": 1.0, "return_decibel_melgram": true, "htk": false, "norm": "slaney"}}, {"class_name": "Normalization2D", "config": {"name": "normalization2d_7", "trainable": true, "dtype": "float32", "int_axis": 0, "str_axis": null, "image_data_format": "channels_last"}}, {"class_name": "Lambda", "config": {"name": "squeeze_last_dim", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABDAAAAcwwAAAB0AKABfABkAaECUwApAk7p/////ykC2gFL2gdzcXVl\nZXplKQHaAXGpAHIFAAAA+j9DOi9Vc2Vycy9zenltby9BcHBEYXRhL0xvY2FsL1RlbXAvaXB5a2Vy\nbmVsXzE5OTk2LzIxNjA0MjA1NjIucHnaCDxsYW1iZGE+GAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_13", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": ["sparse_categorical_crossentropy"], "metrics": ["sparse_categorical_accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 6.399999983841553e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
¢
_inbound_nodes
_outbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"è
_tf_keras_layerÎ{"class_name": "Reshape", "name": "reshape_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_13", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, -1]}}}
Â
_inbound_nodes
dft_real_kernels
dft_imag_kernels
freq2mel
_outbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"Î
_tf_keras_layer´{"class_name": "Melspectrogram", "name": "mel_stft", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 16000]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "mel_stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 16000]}, "dtype": "float32", "n_dft": 1024, "n_hop": 128, "padding": "same", "power_spectrogram": 2.0, "return_decibel_spectrogram": false, "trainable_kernel": false, "image_data_format": "channels_last", "sr": 16000, "n_mels": 80, "fmin": 40.0, "fmax": 8000.0, "trainable_fb": false, "power_melgram": 1.0, "return_decibel_melgram": true, "htk": false, "norm": "slaney"}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 1, 16000]}}
À
 _inbound_nodes
!_outbound_nodes
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"
_tf_keras_layerì{"class_name": "Normalization2D", "name": "normalization2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalization2d_7", "trainable": true, "dtype": "float32", "int_axis": 0, "str_axis": null, "image_data_format": "channels_last"}}

&_inbound_nodes
'_outbound_nodes
(regularization_losses
)trainable_variables
*	variables
+	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Þ
_tf_keras_layerÄ{"class_name": "Lambda", "name": "squeeze_last_dim", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "squeeze_last_dim", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABDAAAAcwwAAAB0AKABfABkAaECUwApAk7p/////ykC2gFL2gdzcXVl\nZXplKQHaAXGpAHIFAAAA+j9DOi9Vc2Vycy9zenltby9BcHBEYXRhL0xvY2FsL1RlbXAvaXB5a2Vy\nbmVsXzE5OTk2LzIxNjA0MjA1NjIucHnaCDxsYW1iZGE+GAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
¨
,cell
-_inbound_nodes
.
state_spec
/_outbound_nodes
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"Ô	
_tf_keras_rnn_layer¶	{"class_name": "SimpleRNN", "name": "simple_rnn_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_13", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 125]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [32, 80, 125]}}

4_inbound_nodes
5_outbound_nodes
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}

:_inbound_nodes

;kernel
<bias
=_outbound_nodes
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+Ç&call_and_return_all_conditional_losses
È__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 5120]}}

B_inbound_nodes

Ckernel
Dbias
E_outbound_nodes
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+É&call_and_return_all_conditional_losses
Ê__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64]}}

J_inbound_nodes

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+Ë&call_and_return_all_conditional_losses
Ì__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32]}}
Ã

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rate
Uiterm m¡m¢;m£<m¤Cm¥Dm¦Km§Lm¨Vm©WmªXm«v¬v­v®;v¯<v°Cv±Dv²Kv³Lv´VvµWv¶Xv·"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
V3
W4
X5
;6
<7
C8
D9
K10
L11"
trackable_list_wrapper
v
0
1
2
V3
W4
X5
;6
<7
C8
D9
K10
L11"
trackable_list_wrapper
Î

Ylayers
Zmetrics
[non_trainable_variables
\layer_regularization_losses
]layer_metrics
regularization_losses
trainable_variables
	variables
º__call__
¸_default_save_signature
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
-
Íserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

^layers
_layer_regularization_losses
`metrics
anon_trainable_variables
blayer_metrics
regularization_losses
trainable_variables
	variables
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
1:/2mel_stft/real_kernels
1:/2mel_stft/imag_kernels
$:"	P2mel_stft/Variable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
°

clayers
dlayer_regularization_losses
emetrics
fnon_trainable_variables
glayer_metrics
regularization_losses
trainable_variables
	variables
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

hlayers
ilayer_regularization_losses
jmetrics
knon_trainable_variables
llayer_metrics
"regularization_losses
#trainable_variables
$	variables
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

mlayers
nlayer_regularization_losses
ometrics
pnon_trainable_variables
qlayer_metrics
(regularization_losses
)trainable_variables
*	variables
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
ë

Vkernel
Wrecurrent_kernel
Xbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
+Î&call_and_return_all_conditional_losses
Ï__call__"®
_tf_keras_layer{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
¼

vlayers
wlayer_regularization_losses
xnon_trainable_variables
ymetrics
zlayer_metrics
0regularization_losses
1trainable_variables

{states
2	variables
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
±

|layers
}layer_regularization_losses
~metrics
non_trainable_variables
layer_metrics
6regularization_losses
7trainable_variables
8	variables
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
": 	(@2dense_37/kernel
:@2dense_37/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
µ
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
>regularization_losses
?trainable_variables
@	variables
È__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:@ 2dense_38/kernel
: 2dense_38/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
µ
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
Fregularization_losses
Gtrainable_variables
H	variables
Ê__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!: $2dense_39/kernel
:$2dense_39/bias
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
µ
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
Mregularization_losses
Ntrainable_variables
O	variables
Ì__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
9:7}@2'simple_rnn_13/simple_rnn_cell_13/kernel
C:A@@21simple_rnn_13/simple_rnn_cell_13/recurrent_kernel
3:1@2%simple_rnn_13/simple_rnn_cell_13/bias
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
µ
layers
 layer_regularization_losses
metrics
non_trainable_variables
layer_metrics
rregularization_losses
strainable_variables
t	variables
Ï__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
±

total

count

_fn_kwargs
	variables
	keras_api"å
_tf_keras_metricÊ{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
6:42Adam/mel_stft/real_kernels/m
6:42Adam/mel_stft/imag_kernels/m
):'	P2Adam/mel_stft/Variable/m
':%	(@2Adam/dense_37/kernel/m
 :@2Adam/dense_37/bias/m
&:$@ 2Adam/dense_38/kernel/m
 : 2Adam/dense_38/bias/m
&:$ $2Adam/dense_39/kernel/m
 :$2Adam/dense_39/bias/m
>:<}@2.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m
H:F@@28Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m
8:6@2,Adam/simple_rnn_13/simple_rnn_cell_13/bias/m
6:42Adam/mel_stft/real_kernels/v
6:42Adam/mel_stft/imag_kernels/v
):'	P2Adam/mel_stft/Variable/v
':%	(@2Adam/dense_37/kernel/v
 :@2Adam/dense_37/bias/v
&:$@ 2Adam/dense_38/kernel/v
 : 2Adam/dense_38/bias/v
&:$ $2Adam/dense_39/kernel/v
 :$2Adam/dense_39/bias/v
>:<}@2.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v
H:F@@28Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v
8:6@2,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v
é2æ
!__inference__wrapped_model_479294À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
reshape_13_inputÿÿÿÿÿÿÿÿÿ}
ò2ï
I__inference_sequential_19_layer_call_and_return_conditional_losses_480753
I__inference_sequential_19_layer_call_and_return_conditional_losses_481486
I__inference_sequential_19_layer_call_and_return_conditional_losses_480978
I__inference_sequential_19_layer_call_and_return_conditional_losses_481261À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_sequential_19_layer_call_fn_481544
.__inference_sequential_19_layer_call_fn_481515
.__inference_sequential_19_layer_call_fn_481007
.__inference_sequential_19_layer_call_fn_481036À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_reshape_13_layer_call_and_return_conditional_losses_481557¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_reshape_13_layer_call_fn_481562¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
D__inference_mel_stft_layer_call_and_return_conditional_losses_481633
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
)__inference_mel_stft_layer_call_fn_481644
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿ2ü
M__inference_normalization2d_7_layer_call_and_return_conditional_losses_481661ª
¡²
FullArgSpec 
args
jself
jx
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ä2á
2__inference_normalization2d_7_layer_call_fn_481666ª
¡²
FullArgSpec 
args
jself
jx
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_481671
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_481676À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¬2©
1__inference_squeeze_last_dim_layer_call_fn_481681
1__inference_squeeze_last_dim_layer_call_fn_481686À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_481798
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_482156
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_482044
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_481910Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_simple_rnn_13_layer_call_fn_481932
.__inference_simple_rnn_13_layer_call_fn_481921
.__inference_simple_rnn_13_layer_call_fn_482167
.__inference_simple_rnn_13_layer_call_fn_482178Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_flatten_8_layer_call_and_return_conditional_losses_482184¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_8_layer_call_fn_482189¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_37_layer_call_and_return_conditional_losses_482200¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_37_layer_call_fn_482209¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_38_layer_call_and_return_conditional_losses_482220¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_38_layer_call_fn_482229¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_39_layer_call_and_return_conditional_losses_482240¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_39_layer_call_fn_482249¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
<B:
$__inference_signature_wrapper_480528reshape_13_input
ä2á
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_482266
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_482283¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
®2«
3__inference_simple_rnn_cell_13_layer_call_fn_482297
3__inference_simple_rnn_cell_13_layer_call_fn_482311¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ¤
!__inference__wrapped_model_479294VXW;<CDKL:¢7
0¢-
+(
reshape_13_inputÿÿÿÿÿÿÿÿÿ}
ª "3ª0
.
dense_39"
dense_39ÿÿÿÿÿÿÿÿÿ$¥
D__inference_dense_37_layer_call_and_return_conditional_losses_482200];<0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ(
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_37_layer_call_fn_482209P;<0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_38_layer_call_and_return_conditional_losses_482220\CD/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 |
)__inference_dense_38_layer_call_fn_482229OCD/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¤
D__inference_dense_39_layer_call_and_return_conditional_losses_482240\KL/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ$
 |
)__inference_dense_39_layer_call_fn_482249OKL/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ$¦
E__inference_flatten_8_layer_call_and_return_conditional_losses_482184]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿP@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ(
 ~
*__inference_flatten_8_layer_call_fn_482189P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿP@
ª "ÿÿÿÿÿÿÿÿÿ(­
D__inference_mel_stft_layer_call_and_return_conditional_losses_481633e/¢,
%¢"
 
xÿÿÿÿÿÿÿÿÿ}
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿP}
 
)__inference_mel_stft_layer_call_fn_481644X/¢,
%¢"
 
xÿÿÿÿÿÿÿÿÿ}
ª " ÿÿÿÿÿÿÿÿÿP}¸
M__inference_normalization2d_7_layer_call_and_return_conditional_losses_481661g6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿP}

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿP}
 
2__inference_normalization2d_7_layer_call_fn_481666Z6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿP}

 
ª " ÿÿÿÿÿÿÿÿÿP}¨
F__inference_reshape_13_layer_call_and_return_conditional_losses_481557^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ}
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ}
 
+__inference_reshape_13_layer_call_fn_481562Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ}
ª "ÿÿÿÿÿÿÿÿÿ}¼
I__inference_sequential_19_layer_call_and_return_conditional_losses_480753oVXW;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ}
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ$
 ¼
I__inference_sequential_19_layer_call_and_return_conditional_losses_480978oVXW;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ}
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ$
 Æ
I__inference_sequential_19_layer_call_and_return_conditional_losses_481261yVXW;<CDKLB¢?
8¢5
+(
reshape_13_inputÿÿÿÿÿÿÿÿÿ}
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ$
 Æ
I__inference_sequential_19_layer_call_and_return_conditional_losses_481486yVXW;<CDKLB¢?
8¢5
+(
reshape_13_inputÿÿÿÿÿÿÿÿÿ}
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ$
 
.__inference_sequential_19_layer_call_fn_481007bVXW;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ}
p

 
ª "ÿÿÿÿÿÿÿÿÿ$
.__inference_sequential_19_layer_call_fn_481036bVXW;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ}
p 

 
ª "ÿÿÿÿÿÿÿÿÿ$
.__inference_sequential_19_layer_call_fn_481515lVXW;<CDKLB¢?
8¢5
+(
reshape_13_inputÿÿÿÿÿÿÿÿÿ}
p

 
ª "ÿÿÿÿÿÿÿÿÿ$
.__inference_sequential_19_layer_call_fn_481544lVXW;<CDKLB¢?
8¢5
+(
reshape_13_inputÿÿÿÿÿÿÿÿÿ}
p 

 
ª "ÿÿÿÿÿÿÿÿÿ$¼
$__inference_signature_wrapper_480528VXW;<CDKLN¢K
¢ 
DªA
?
reshape_13_input+(
reshape_13_inputÿÿÿÿÿÿÿÿÿ}"3ª0
.
dense_39"
dense_39ÿÿÿÿÿÿÿÿÿ$Ø
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_481798VXWO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ø
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_481910VXWO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¾
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_482044qVXW?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿP}

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿP@
 ¾
I__inference_simple_rnn_13_layer_call_and_return_conditional_losses_482156qVXW?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿP}

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿP@
 ¯
.__inference_simple_rnn_13_layer_call_fn_481921}VXWO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¯
.__inference_simple_rnn_13_layer_call_fn_481932}VXWO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
.__inference_simple_rnn_13_layer_call_fn_482167dVXW?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿP}

 
p

 
ª "ÿÿÿÿÿÿÿÿÿP@
.__inference_simple_rnn_13_layer_call_fn_482178dVXW?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿP}

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿP@
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_482266·VXW\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ}
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 
N__inference_simple_rnn_cell_13_layer_call_and_return_conditional_losses_482283·VXW\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ}
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 á
3__inference_simple_rnn_cell_13_layer_call_fn_482297©VXW\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ}
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@á
3__inference_simple_rnn_cell_13_layer_call_fn_482311©VXW\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ}
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@¼
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_481671l?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿP}

 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿP}
 ¼
L__inference_squeeze_last_dim_layer_call_and_return_conditional_losses_481676l?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿP}

 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿP}
 
1__inference_squeeze_last_dim_layer_call_fn_481681_?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿP}

 
p
ª "ÿÿÿÿÿÿÿÿÿP}
1__inference_squeeze_last_dim_layer_call_fn_481686_?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿP}

 
p 
ª "ÿÿÿÿÿÿÿÿÿP}