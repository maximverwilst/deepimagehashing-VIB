??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
*
Erf
x"T
y"T"
Ttype:
2
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
?
tbh/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_nametbh/dense_8/kernel
y
&tbh/dense_8/kernel/Read/ReadVariableOpReadVariableOptbh/dense_8/kernel*
_output_shapes

:@*
dtype0
x
tbh/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametbh/dense_8/bias
q
$tbh/dense_8/bias/Read/ReadVariableOpReadVariableOptbh/dense_8/bias*
_output_shapes
:*
dtype0
?
tbh/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_nametbh/dense_9/kernel
z
&tbh/dense_9/kernel/Read/ReadVariableOpReadVariableOptbh/dense_9/kernel*
_output_shapes
:	?*
dtype0
x
tbh/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametbh/dense_9/bias
q
$tbh/dense_9/bias/Read/ReadVariableOpReadVariableOptbh/dense_9/bias*
_output_shapes
:*
dtype0
?
!tbh/vae_encoder_geco/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?*2
shared_name#!tbh/vae_encoder_geco/dense/kernel
?
5tbh/vae_encoder_geco/dense/kernel/Read/ReadVariableOpReadVariableOp!tbh/vae_encoder_geco/dense/kernel* 
_output_shapes
:
?
?*
dtype0
?
tbh/vae_encoder_geco/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!tbh/vae_encoder_geco/dense/bias
?
3tbh/vae_encoder_geco/dense/bias/Read/ReadVariableOpReadVariableOptbh/vae_encoder_geco/dense/bias*
_output_shapes	
:?*
dtype0
?
#tbh/vae_encoder_geco/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#tbh/vae_encoder_geco/dense_1/kernel
?
7tbh/vae_encoder_geco/dense_1/kernel/Read/ReadVariableOpReadVariableOp#tbh/vae_encoder_geco/dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
!tbh/vae_encoder_geco/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!tbh/vae_encoder_geco/dense_1/bias
?
5tbh/vae_encoder_geco/dense_1/bias/Read/ReadVariableOpReadVariableOp!tbh/vae_encoder_geco/dense_1/bias*
_output_shapes	
:?*
dtype0
?
#tbh/vae_encoder_geco/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#tbh/vae_encoder_geco/dense_2/kernel
?
7tbh/vae_encoder_geco/dense_2/kernel/Read/ReadVariableOpReadVariableOp#tbh/vae_encoder_geco/dense_2/kernel* 
_output_shapes
:
??*
dtype0
?
!tbh/vae_encoder_geco/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!tbh/vae_encoder_geco/dense_2/bias
?
5tbh/vae_encoder_geco/dense_2/bias/Read/ReadVariableOpReadVariableOp!tbh/vae_encoder_geco/dense_2/bias*
_output_shapes	
:?*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
?
?*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
??
*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?
*
dtype0
?
tbh/decoder/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nametbh/decoder/dense_5/kernel
?
.tbh/decoder/dense_5/kernel/Read/ReadVariableOpReadVariableOptbh/decoder/dense_5/kernel* 
_output_shapes
:
??*
dtype0
?
tbh/decoder/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nametbh/decoder/dense_5/bias
?
,tbh/decoder/dense_5/bias/Read/ReadVariableOpReadVariableOptbh/decoder/dense_5/bias*
_output_shapes	
:?*
dtype0
?
tbh/decoder/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*+
shared_nametbh/decoder/dense_6/kernel
?
.tbh/decoder/dense_6/kernel/Read/ReadVariableOpReadVariableOptbh/decoder/dense_6/kernel* 
_output_shapes
:
??
*
dtype0
?
tbh/decoder/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*)
shared_nametbh/decoder/dense_6/bias
?
,tbh/decoder/dense_6/bias/Read/ReadVariableOpReadVariableOptbh/decoder/dense_6/bias*
_output_shapes	
:?
*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
??*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?9 B?9
?
encoder
decoder
tbn
	dis_1
	dis_2
trainable_variables
	variables
regularization_losses
		keras_api


signatures
?
fc_1

fc_2_1

fc_2_2
reconstruction1
reconstruction2
trainable_variables
	variables
regularization_losses
	keras_api
f
fc_1
fc_2
trainable_variables
	variables
regularization_losses
	keras_api
[
gcn
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
16
 17
%18
&19
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
16
 17
%18
&19
 
?
;layer_regularization_losses
<metrics
=non_trainable_variables
>layer_metrics
trainable_variables

?layers
	variables
regularization_losses
 
h

+kernel
,bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
h

-kernel
.bias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
h

/kernel
0bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
h

1kernel
2bias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
h

3kernel
4bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
F
+0
,1
-2
.3
/4
05
16
27
38
49
F
+0
,1
-2
.3
/4
05
16
27
38
49
 
?
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
Wnon_trainable_variables
trainable_variables

Xlayers
	variables
regularization_losses
h

5kernel
6bias
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
h

7kernel
8bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api

50
61
72
83

50
61
72
83
 
?
alayer_regularization_losses
bmetrics
clayer_metrics
dnon_trainable_variables
trainable_variables

elayers
	variables
regularization_losses
b
ffc
grs
htrainable_variables
i	variables
jregularization_losses
k	keras_api

90
:1

90
:1
 
?
llayer_regularization_losses
mmetrics
nlayer_metrics
onon_trainable_variables
trainable_variables

players
	variables
regularization_losses
OM
VARIABLE_VALUEtbh/dense_8/kernel'dis_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtbh/dense_8/bias%dis_1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
qlayer_regularization_losses
rmetrics
slayer_metrics
tnon_trainable_variables
!trainable_variables

ulayers
"	variables
#regularization_losses
OM
VARIABLE_VALUEtbh/dense_9/kernel'dis_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtbh/dense_9/bias%dis_2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
vlayer_regularization_losses
wmetrics
xlayer_metrics
ynon_trainable_variables
'trainable_variables

zlayers
(	variables
)regularization_losses
ge
VARIABLE_VALUE!tbh/vae_encoder_geco/dense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtbh/vae_encoder_geco/dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#tbh/vae_encoder_geco/dense_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!tbh/vae_encoder_geco/dense_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#tbh/vae_encoder_geco/dense_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!tbh/vae_encoder_geco/dense_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtbh/decoder/dense_5/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEtbh/decoder/dense_5/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtbh/decoder/dense_6/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEtbh/decoder/dense_6/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_7/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_7/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
#
0
1
2
3
4

+0
,1

+0
,1
 
?
{layer_regularization_losses
|metrics
}layer_metrics
~non_trainable_variables
@trainable_variables

layers
A	variables
Bregularization_losses

-0
.1

-0
.1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Dtrainable_variables
?layers
E	variables
Fregularization_losses

/0
01

/0
01
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Htrainable_variables
?layers
I	variables
Jregularization_losses

10
21

10
21
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Ltrainable_variables
?layers
M	variables
Nregularization_losses

30
41

30
41
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Ptrainable_variables
?layers
Q	variables
Rregularization_losses
 
 
 
 
#
0
1
2
3
4

50
61

50
61
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Ytrainable_variables
?layers
Z	variables
[regularization_losses

70
81

70
81
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
]trainable_variables
?layers
^	variables
_regularization_losses
 
 
 
 

0
1
l

9kernel
:bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

90
:1

90
:1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
htrainable_variables
?layers
i	variables
jregularization_losses
 
 
 
 

0
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

90
:1

90
:1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?trainable_variables
?layers
?	variables
?regularization_losses
 
 
 
 
 

f0
g1
 
 
 
 
 
t
serving_default_input_1_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_input_1_2Placeholder*(
_output_shapes
:??????????
*
dtype0*
shape:??????????

|
serving_default_input_1_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????@*
dtype0*
shape:?????????@
|
serving_default_input_3Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1_1serving_default_input_1_2serving_default_input_1_3serving_default_input_2serving_default_input_3!tbh/vae_encoder_geco/dense/kerneltbh/vae_encoder_geco/dense/bias#tbh/vae_encoder_geco/dense_1/kernel!tbh/vae_encoder_geco/dense_1/bias#tbh/vae_encoder_geco/dense_2/kernel!tbh/vae_encoder_geco/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_334898718
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&tbh/dense_8/kernel/Read/ReadVariableOp$tbh/dense_8/bias/Read/ReadVariableOp&tbh/dense_9/kernel/Read/ReadVariableOp$tbh/dense_9/bias/Read/ReadVariableOp5tbh/vae_encoder_geco/dense/kernel/Read/ReadVariableOp3tbh/vae_encoder_geco/dense/bias/Read/ReadVariableOp7tbh/vae_encoder_geco/dense_1/kernel/Read/ReadVariableOp5tbh/vae_encoder_geco/dense_1/bias/Read/ReadVariableOp7tbh/vae_encoder_geco/dense_2/kernel/Read/ReadVariableOp5tbh/vae_encoder_geco/dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp.tbh/decoder/dense_5/kernel/Read/ReadVariableOp,tbh/decoder/dense_5/bias/Read/ReadVariableOp.tbh/decoder/dense_6/kernel/Read/ReadVariableOp,tbh/decoder/dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpConst*!
Tin
2*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_save_334899995
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametbh/dense_8/kerneltbh/dense_8/biastbh/dense_9/kerneltbh/dense_9/bias!tbh/vae_encoder_geco/dense/kerneltbh/vae_encoder_geco/dense/bias#tbh/vae_encoder_geco/dense_1/kernel!tbh/vae_encoder_geco/dense_1/bias#tbh/vae_encoder_geco/dense_2/kernel!tbh/vae_encoder_geco/dense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biastbh/decoder/dense_5/kerneltbh/decoder/dense_5/biastbh/decoder/dense_6/kerneltbh/decoder/dense_6/biasdense_7/kerneldense_7/bias* 
Tin
2*
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
GPU2*0J 8? *.
f)R'
%__inference__traced_restore_334900065??
?	
?
F__inference_dense_9_layer_call_and_return_conditional_losses_334898502

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_8_layer_call_and_return_conditional_losses_334899839

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:@
 
_user_specified_nameinputs
?	
?
F__inference_dense_9_layer_call_and_return_conditional_losses_334898352

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?^
?
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334898209

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddi
dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu/mul/x?
dense/Gelu/mulMuldense/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mulk
dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu/Cast/x?
dense/Gelu/truedivRealDivdense/BiasAdd:output:0dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/truedivr
dense/Gelu/ErfErfdense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/Erfi
dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu/add/x?
dense/Gelu/addAddV2dense/Gelu/add/x:output:0dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/add?
dense/Gelu/mul_1Muldense/Gelu/mul:z:0dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mul_1?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1m
dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu_1/mul/x?
dense/Gelu_1/mulMuldense/Gelu_1/mul/x:output:0dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mulo
dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu_1/Cast/x?
dense/Gelu_1/truedivRealDivdense/BiasAdd_1:output:0dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/truedivx
dense/Gelu_1/ErfErfdense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/Erfm
dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu_1/add/x?
dense/Gelu_1/addAddV2dense/Gelu_1/add/x:output:0dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/add?
dense/Gelu_1/mul_1Muldense/Gelu_1/mul:z:0dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Gelu_1/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1c
zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
zerosL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1^
mulMulzeros:output:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-334898158*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Gelu/mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/SigmoidT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2	
split_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_2/shapew
	Reshape_2Reshapesplit_1:output:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_2s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_3/shapew
	Reshape_3Reshapesplit_1:output:1Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_3i
zeros_1Const*
_output_shapes
:	?*
dtype0*
valueB	?*    2	
zeros_1e
mul_1MulReshape_3:output:0zeros_1:output:0*
T0*
_output_shapes
:	?2
mul_1`
add_4AddV2Reshape_2:output:0	mul_1:z:0*
T0*
_output_shapes
:	?2
add_4?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identity	add_4:z:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????
::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?"
?
F__inference_decoder_layer_call_and_return_conditional_losses_334898400

inputs*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/BiasAddm
dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_5/Gelu/mul/x?
dense_5/Gelu/mulMuldense_5/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mulo
dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_5/Gelu/Cast/x?
dense_5/Gelu/truedivRealDivdense_5/BiasAdd:output:0dense_5/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/truedivo
dense_5/Gelu/ErfErfdense_5/Gelu/truediv:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/Erfm
dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_5/Gelu/add/x?
dense_5/Gelu/addAddV2dense_5/Gelu/add/x:output:0dense_5/Gelu/Erf:y:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/add?
dense_5/Gelu/mul_1Muldense_5/Gelu/mul:z:0dense_5/Gelu/add:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mul_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/Gelu/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?
*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/BiasAddm
dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_6/Gelu/mul/x?
dense_6/Gelu/mulMuldense_6/Gelu/mul/x:output:0dense_6/BiasAdd:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mulo
dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_6/Gelu/Cast/x?
dense_6/Gelu/truedivRealDivdense_6/BiasAdd:output:0dense_6/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/truedivo
dense_6/Gelu/ErfErfdense_6/Gelu/truediv:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/Erfm
dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_6/Gelu/add/x?
dense_6/Gelu/addAddV2dense_6/Gelu/add/x:output:0dense_6/Gelu/Erf:y:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/add?
dense_6/Gelu/mul_1Muldense_6/Gelu/mul:z:0dense_6/Gelu/add:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mul_1?
IdentityIdentitydense_6/Gelu/mul_1:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?
2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
'__inference_tbh_layer_call_fn_334899417

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_2
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1
inputs_0_2inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14* 
Tin
2*
Tout

2*
_collective_manager_ids
 *c
_output_shapesQ
O:@:	?
:::?????????:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_3348986072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	?
2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:??????????
:?????????:?????????@:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:??????????

$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?	
?
F__inference_dense_8_layer_call_and_return_conditional_losses_334898325

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:@
 
_user_specified_nameinputs
?
?
3__inference_twin_bottleneck_layer_call_fn_334899818
bbn
cbn
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbbncbnunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_3348982872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:@:	?::22
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:D@

_output_shapes
:	?

_user_specified_namecbn
?
I
*__inference_build_adjacency_hamming_820293
	tensor_in
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CastS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/yU
subSub	tensor_insub/y:output:0*
T0*
_output_shapes

:@2
subj
MatMulMatMul	tensor_insub:z:0*
T0*
_output_shapes

:*
transpose_b(2
MatMuln
MatMul_1MatMulsub:z:0	tensor_in*
T0*
_output_shapes

:*
transpose_b(2

MatMul_1b
addAddV2MatMul:product:0MatMul_1:product:0*
T0*
_output_shapes

:2
addC
AbsAbsadd:z:0*
T0*
_output_shapes

:2
AbsY
truedivRealDivAbs:y:0Cast:y:0*
T0*
_output_shapes

:2	
truedivW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/x]
sub_1Subsub_1/x:output:0truediv:z:0*
T0*
_output_shapes

:2
sub_1S
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *33??2
Pow/yU
PowPow	sub_1:z:0Pow/y:output:0*
T0*
_output_shapes

:2
PowR
IdentityIdentityPow:z:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

:@:I E

_output_shapes

:@
#
_user_specified_name	tensor_in
?r
?
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334898109

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddi
dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu/mul/x?
dense/Gelu/mulMuldense/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mulk
dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu/Cast/x?
dense/Gelu/truedivRealDivdense/BiasAdd:output:0dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/truedivr
dense/Gelu/ErfErfdense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/Erfi
dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu/add/x?
dense/Gelu/addAddV2dense/Gelu/add/x:output:0dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/add?
dense/Gelu/mul_1Muldense/Gelu/mul:z:0dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mul_1?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1m
dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu_1/mul/x?
dense/Gelu_1/mulMuldense/Gelu_1/mul/x:output:0dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mulo
dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu_1/Cast/x?
dense/Gelu_1/truedivRealDivdense/BiasAdd_1:output:0dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/truedivx
dense/Gelu_1/ErfErfdense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/Erfm
dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu_1/add/x?
dense/Gelu_1/addAddV2dense/Gelu_1/add/x:output:0dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/add?
dense/Gelu_1/mul_1Muldense/Gelu_1/mul:z:0dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Gelu_1/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:@2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@2
random_normalw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumrandom_normal:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:@2
clip_by_valueL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1a
mulMulclip_by_value:z:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-334898049*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Gelu/mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/SigmoidT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2	
split_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_2/shapew
	Reshape_2Reshapesplit_1:output:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_2s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_3/shapew
	Reshape_3Reshapesplit_1:output:1Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_3
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:	?*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*
_output_shapes
:	?2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:	?2
random_normal_1{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimumrandom_normal_1:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:	?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:	?2
clip_by_value_1h
mul_1MulReshape_3:output:0clip_by_value_1:z:0*
T0*
_output_shapes
:	?2
mul_1`
add_4AddV2Reshape_2:output:0	mul_1:z:0*
T0*
_output_shapes
:	?2
add_4?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identity	add_4:z:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????
::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
??
?
B__inference_tbh_layer_call_and_return_conditional_losses_334899366

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_29
5vae_encoder_geco_dense_matmul_readvariableop_resource:
6vae_encoder_geco_dense_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_1_matmul_readvariableop_resource<
8vae_encoder_geco_dense_1_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_2_matmul_readvariableop_resource<
8vae_encoder_geco_dense_2_biasadd_readvariableop_resource
identity??-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?,vae_encoder_geco/dense/MatMul/ReadVariableOp?.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_2/MatMul/ReadVariableOpj
vae_encoder_geco/ShapeShape
inputs_0_1*
T0*
_output_shapes
:2
vae_encoder_geco/Shape?
$vae_encoder_geco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$vae_encoder_geco/strided_slice/stack?
&vae_encoder_geco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_1?
&vae_encoder_geco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_2?
vae_encoder_geco/strided_sliceStridedSlicevae_encoder_geco/Shape:output:0-vae_encoder_geco/strided_slice/stack:output:0/vae_encoder_geco/strided_slice/stack_1:output:0/vae_encoder_geco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
vae_encoder_geco/strided_slice?
,vae_encoder_geco/dense/MatMul/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02.
,vae_encoder_geco/dense/MatMul/ReadVariableOp?
vae_encoder_geco/dense/MatMulMatMul
inputs_0_14vae_encoder_geco/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder_geco/dense/MatMul?
-vae_encoder_geco/dense/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?
vae_encoder_geco/dense/BiasAddBiasAdd'vae_encoder_geco/dense/MatMul:product:05vae_encoder_geco/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
vae_encoder_geco/dense/BiasAdd?
!vae_encoder_geco/dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!vae_encoder_geco/dense/Gelu/mul/x?
vae_encoder_geco/dense/Gelu/mulMul*vae_encoder_geco/dense/Gelu/mul/x:output:0'vae_encoder_geco/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/mul?
"vae_encoder_geco/dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2$
"vae_encoder_geco/dense/Gelu/Cast/x?
#vae_encoder_geco/dense/Gelu/truedivRealDiv'vae_encoder_geco/dense/BiasAdd:output:0+vae_encoder_geco/dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu/truediv?
vae_encoder_geco/dense/Gelu/ErfErf'vae_encoder_geco/dense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/Erf?
!vae_encoder_geco/dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!vae_encoder_geco/dense/Gelu/add/x?
vae_encoder_geco/dense/Gelu/addAddV2*vae_encoder_geco/dense/Gelu/add/x:output:0#vae_encoder_geco/dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/add?
!vae_encoder_geco/dense/Gelu/mul_1Mul#vae_encoder_geco/dense/Gelu/mul:z:0#vae_encoder_geco/dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu/mul_1?
.vae_encoder_geco/dense/MatMul_1/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype020
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?
vae_encoder_geco/dense/MatMul_1MatMul
inputs_0_16vae_encoder_geco/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/MatMul_1?
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?
 vae_encoder_geco/dense/BiasAdd_1BiasAdd)vae_encoder_geco/dense/MatMul_1:product:07vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense/BiasAdd_1?
#vae_encoder_geco/dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#vae_encoder_geco/dense/Gelu_1/mul/x?
!vae_encoder_geco/dense/Gelu_1/mulMul,vae_encoder_geco/dense/Gelu_1/mul/x:output:0)vae_encoder_geco/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/mul?
$vae_encoder_geco/dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2&
$vae_encoder_geco/dense/Gelu_1/Cast/x?
%vae_encoder_geco/dense/Gelu_1/truedivRealDiv)vae_encoder_geco/dense/BiasAdd_1:output:0-vae_encoder_geco/dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2'
%vae_encoder_geco/dense/Gelu_1/truediv?
!vae_encoder_geco/dense/Gelu_1/ErfErf)vae_encoder_geco/dense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/Erf?
#vae_encoder_geco/dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#vae_encoder_geco/dense/Gelu_1/add/x?
!vae_encoder_geco/dense/Gelu_1/addAddV2,vae_encoder_geco/dense/Gelu_1/add/x:output:0%vae_encoder_geco/dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/add?
#vae_encoder_geco/dense/Gelu_1/mul_1Mul%vae_encoder_geco/dense/Gelu_1/mul:z:0%vae_encoder_geco/dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu_1/mul_1?
.vae_encoder_geco/dense_1/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?
vae_encoder_geco/dense_1/MatMulMatMul'vae_encoder_geco/dense/Gelu_1/mul_1:z:06vae_encoder_geco/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_1/MatMul?
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_1/BiasAddBiasAdd)vae_encoder_geco/dense_1/MatMul:product:07vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_1/BiasAddr
vae_encoder_geco/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const?
 vae_encoder_geco/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 vae_encoder_geco/split/split_dim?
vae_encoder_geco/splitSplit)vae_encoder_geco/split/split_dim:output:0)vae_encoder_geco/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder_geco/split?
vae_encoder_geco/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2 
vae_encoder_geco/Reshape/shape?
vae_encoder_geco/ReshapeReshapevae_encoder_geco/split:output:0'vae_encoder_geco/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape?
 vae_encoder_geco/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2"
 vae_encoder_geco/Reshape_1/shape?
vae_encoder_geco/Reshape_1Reshapevae_encoder_geco/split:output:1)vae_encoder_geco/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape_1?
vae_encoder_geco/zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
vae_encoder_geco/zeros
vae_encoder_geco/NegNeg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Negv
vae_encoder_geco/ExpExpvae_encoder_geco/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp?
vae_encoder_geco/Neg_1Neg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_1|
vae_encoder_geco/Exp_1Expvae_encoder_geco/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_1u
vae_encoder_geco/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add/y?
vae_encoder_geco/addAddV2vae_encoder_geco/Exp_1:y:0vae_encoder_geco/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/addu
vae_encoder_geco/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow/y?
vae_encoder_geco/powPowvae_encoder_geco/add:z:0vae_encoder_geco/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow?
vae_encoder_geco/truedivRealDivvae_encoder_geco/Exp:y:0vae_encoder_geco/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv?
vae_encoder_geco/Neg_2Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_2|
vae_encoder_geco/Exp_2Expvae_encoder_geco/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_2?
vae_encoder_geco/Neg_3Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_3|
vae_encoder_geco/Exp_3Expvae_encoder_geco/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_3y
vae_encoder_geco/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_1/y?
vae_encoder_geco/add_1AddV2vae_encoder_geco/Exp_3:y:0!vae_encoder_geco/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_1y
vae_encoder_geco/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow_1/y?
vae_encoder_geco/pow_1Powvae_encoder_geco/add_1:z:0!vae_encoder_geco/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow_1?
vae_encoder_geco/truediv_1RealDivvae_encoder_geco/Exp_2:y:0vae_encoder_geco/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_1?
vae_encoder_geco/mulMulvae_encoder_geco/zeros:output:0#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/mul?
vae_encoder_geco/add_2AddV2vae_encoder_geco/mul:z:0!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_2{
vae_encoder_geco/SignSignvae_encoder_geco/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Signy
vae_encoder_geco/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_3/y?
vae_encoder_geco/add_3AddV2vae_encoder_geco/Sign:y:0!vae_encoder_geco/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_3?
vae_encoder_geco/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/truediv_2/y?
vae_encoder_geco/truediv_2RealDivvae_encoder_geco/add_3:z:0%vae_encoder_geco/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_2?
vae_encoder_geco/IdentityIdentityvae_encoder_geco/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Identity?
vae_encoder_geco/IdentityN	IdentityNvae_encoder_geco/truediv_2:z:0!vae_encoder_geco/Reshape:output:0#vae_encoder_geco/Reshape_1:output:0vae_encoder_geco/zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-334899316*<
_output_shapes*
(:@:@:@:@2
vae_encoder_geco/IdentityN?
.vae_encoder_geco/dense_2/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp?
vae_encoder_geco/dense_2/MatMulMatMul%vae_encoder_geco/dense/Gelu/mul_1:z:06vae_encoder_geco/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_2/MatMul?
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_2/BiasAddBiasAdd)vae_encoder_geco/dense_2/MatMul:product:07vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/BiasAdd?
 vae_encoder_geco/dense_2/SigmoidSigmoid)vae_encoder_geco/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/Sigmoidv
vae_encoder_geco/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const_1?
"vae_encoder_geco/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"vae_encoder_geco/split_1/split_dim?
vae_encoder_geco/split_1Split+vae_encoder_geco/split_1/split_dim:output:0$vae_encoder_geco/dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2
vae_encoder_geco/split_1?
 vae_encoder_geco/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_2/shape?
vae_encoder_geco/Reshape_2Reshape!vae_encoder_geco/split_1:output:0)vae_encoder_geco/Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_2?
 vae_encoder_geco/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_3/shape?
vae_encoder_geco/Reshape_3Reshape!vae_encoder_geco/split_1:output:1)vae_encoder_geco/Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_3?
vae_encoder_geco/zeros_1Const*
_output_shapes
:	?*
dtype0*
valueB	?*    2
vae_encoder_geco/zeros_1?
vae_encoder_geco/mul_1Mul#vae_encoder_geco/Reshape_3:output:0!vae_encoder_geco/zeros_1:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/mul_1?
vae_encoder_geco/add_4AddV2#vae_encoder_geco/Reshape_2:output:0vae_encoder_geco/mul_1:z:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/add_4?
IdentityIdentity#vae_encoder_geco/IdentityN:output:0.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:??????????
:?????????:?????????@:??????????::::::2^
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp-vae_encoder_geco/dense/BiasAdd/ReadVariableOp2b
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp2\
,vae_encoder_geco/dense/MatMul/ReadVariableOp,vae_encoder_geco/dense/MatMul/ReadVariableOp2`
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp.vae_encoder_geco/dense/MatMul_1/ReadVariableOp2b
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp.vae_encoder_geco/dense_1/MatMul/ReadVariableOp2b
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp.vae_encoder_geco/dense_2/MatMul/ReadVariableOp:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:??????????

$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?	
?
F__inference_dense_9_layer_call_and_return_conditional_losses_334899899

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
+__inference_decoder_layer_call_fn_334899771

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_decoder_layer_call_and_return_conditional_losses_3348984002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?
2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
3__inference_twin_bottleneck_layer_call_fn_334899828
bbn
cbn
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbbncbnunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_3348982872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:@:	?::22
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:D@

_output_shapes
:	?

_user_specified_namecbn
?
?
B__inference_tbh_layer_call_and_return_conditional_losses_334898680

inputs
inputs_1
inputs_2
inputs_3
inputs_4
vae_encoder_geco_334898665
vae_encoder_geco_334898667
vae_encoder_geco_334898669
vae_encoder_geco_334898671
vae_encoder_geco_334898673
vae_encoder_geco_334898675
identity??(vae_encoder_geco/StatefulPartitionedCall?
(vae_encoder_geco/StatefulPartitionedCallStatefulPartitionedCallinputs_1vae_encoder_geco_334898665vae_encoder_geco_334898667vae_encoder_geco_334898669vae_encoder_geco_334898671vae_encoder_geco_334898673vae_encoder_geco_334898675*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:@:	?*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_3348982092*
(vae_encoder_geco/StatefulPartitionedCall?
IdentityIdentity1vae_encoder_geco/StatefulPartitionedCall:output:0)^vae_encoder_geco/StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:??????????
:?????????:?????????@:??????????::::::2T
(vae_encoder_geco/StatefulPartitionedCall(vae_encoder_geco/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
B__inference_tbh_layer_call_and_return_conditional_losses_334899006
	input_1_1
	input_1_2
	input_1_3
input_2
input_39
5vae_encoder_geco_dense_matmul_readvariableop_resource:
6vae_encoder_geco_dense_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_1_matmul_readvariableop_resource<
8vae_encoder_geco_dense_1_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_2_matmul_readvariableop_resource<
8vae_encoder_geco_dense_2_biasadd_readvariableop_resource
identity??-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?,vae_encoder_geco/dense/MatMul/ReadVariableOp?.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_2/MatMul/ReadVariableOpi
vae_encoder_geco/ShapeShape	input_1_2*
T0*
_output_shapes
:2
vae_encoder_geco/Shape?
$vae_encoder_geco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$vae_encoder_geco/strided_slice/stack?
&vae_encoder_geco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_1?
&vae_encoder_geco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_2?
vae_encoder_geco/strided_sliceStridedSlicevae_encoder_geco/Shape:output:0-vae_encoder_geco/strided_slice/stack:output:0/vae_encoder_geco/strided_slice/stack_1:output:0/vae_encoder_geco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
vae_encoder_geco/strided_slice?
,vae_encoder_geco/dense/MatMul/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02.
,vae_encoder_geco/dense/MatMul/ReadVariableOp?
vae_encoder_geco/dense/MatMulMatMul	input_1_24vae_encoder_geco/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder_geco/dense/MatMul?
-vae_encoder_geco/dense/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?
vae_encoder_geco/dense/BiasAddBiasAdd'vae_encoder_geco/dense/MatMul:product:05vae_encoder_geco/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
vae_encoder_geco/dense/BiasAdd?
!vae_encoder_geco/dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!vae_encoder_geco/dense/Gelu/mul/x?
vae_encoder_geco/dense/Gelu/mulMul*vae_encoder_geco/dense/Gelu/mul/x:output:0'vae_encoder_geco/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/mul?
"vae_encoder_geco/dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2$
"vae_encoder_geco/dense/Gelu/Cast/x?
#vae_encoder_geco/dense/Gelu/truedivRealDiv'vae_encoder_geco/dense/BiasAdd:output:0+vae_encoder_geco/dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu/truediv?
vae_encoder_geco/dense/Gelu/ErfErf'vae_encoder_geco/dense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/Erf?
!vae_encoder_geco/dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!vae_encoder_geco/dense/Gelu/add/x?
vae_encoder_geco/dense/Gelu/addAddV2*vae_encoder_geco/dense/Gelu/add/x:output:0#vae_encoder_geco/dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/add?
!vae_encoder_geco/dense/Gelu/mul_1Mul#vae_encoder_geco/dense/Gelu/mul:z:0#vae_encoder_geco/dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu/mul_1?
.vae_encoder_geco/dense/MatMul_1/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype020
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?
vae_encoder_geco/dense/MatMul_1MatMul	input_1_26vae_encoder_geco/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/MatMul_1?
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?
 vae_encoder_geco/dense/BiasAdd_1BiasAdd)vae_encoder_geco/dense/MatMul_1:product:07vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense/BiasAdd_1?
#vae_encoder_geco/dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#vae_encoder_geco/dense/Gelu_1/mul/x?
!vae_encoder_geco/dense/Gelu_1/mulMul,vae_encoder_geco/dense/Gelu_1/mul/x:output:0)vae_encoder_geco/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/mul?
$vae_encoder_geco/dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2&
$vae_encoder_geco/dense/Gelu_1/Cast/x?
%vae_encoder_geco/dense/Gelu_1/truedivRealDiv)vae_encoder_geco/dense/BiasAdd_1:output:0-vae_encoder_geco/dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2'
%vae_encoder_geco/dense/Gelu_1/truediv?
!vae_encoder_geco/dense/Gelu_1/ErfErf)vae_encoder_geco/dense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/Erf?
#vae_encoder_geco/dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#vae_encoder_geco/dense/Gelu_1/add/x?
!vae_encoder_geco/dense/Gelu_1/addAddV2,vae_encoder_geco/dense/Gelu_1/add/x:output:0%vae_encoder_geco/dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/add?
#vae_encoder_geco/dense/Gelu_1/mul_1Mul%vae_encoder_geco/dense/Gelu_1/mul:z:0%vae_encoder_geco/dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu_1/mul_1?
.vae_encoder_geco/dense_1/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?
vae_encoder_geco/dense_1/MatMulMatMul'vae_encoder_geco/dense/Gelu_1/mul_1:z:06vae_encoder_geco/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_1/MatMul?
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_1/BiasAddBiasAdd)vae_encoder_geco/dense_1/MatMul:product:07vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_1/BiasAddr
vae_encoder_geco/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const?
 vae_encoder_geco/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 vae_encoder_geco/split/split_dim?
vae_encoder_geco/splitSplit)vae_encoder_geco/split/split_dim:output:0)vae_encoder_geco/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder_geco/split?
vae_encoder_geco/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2 
vae_encoder_geco/Reshape/shape?
vae_encoder_geco/ReshapeReshapevae_encoder_geco/split:output:0'vae_encoder_geco/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape?
 vae_encoder_geco/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2"
 vae_encoder_geco/Reshape_1/shape?
vae_encoder_geco/Reshape_1Reshapevae_encoder_geco/split:output:1)vae_encoder_geco/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape_1?
vae_encoder_geco/zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
vae_encoder_geco/zeros
vae_encoder_geco/NegNeg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Negv
vae_encoder_geco/ExpExpvae_encoder_geco/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp?
vae_encoder_geco/Neg_1Neg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_1|
vae_encoder_geco/Exp_1Expvae_encoder_geco/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_1u
vae_encoder_geco/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add/y?
vae_encoder_geco/addAddV2vae_encoder_geco/Exp_1:y:0vae_encoder_geco/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/addu
vae_encoder_geco/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow/y?
vae_encoder_geco/powPowvae_encoder_geco/add:z:0vae_encoder_geco/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow?
vae_encoder_geco/truedivRealDivvae_encoder_geco/Exp:y:0vae_encoder_geco/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv?
vae_encoder_geco/Neg_2Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_2|
vae_encoder_geco/Exp_2Expvae_encoder_geco/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_2?
vae_encoder_geco/Neg_3Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_3|
vae_encoder_geco/Exp_3Expvae_encoder_geco/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_3y
vae_encoder_geco/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_1/y?
vae_encoder_geco/add_1AddV2vae_encoder_geco/Exp_3:y:0!vae_encoder_geco/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_1y
vae_encoder_geco/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow_1/y?
vae_encoder_geco/pow_1Powvae_encoder_geco/add_1:z:0!vae_encoder_geco/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow_1?
vae_encoder_geco/truediv_1RealDivvae_encoder_geco/Exp_2:y:0vae_encoder_geco/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_1?
vae_encoder_geco/mulMulvae_encoder_geco/zeros:output:0#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/mul?
vae_encoder_geco/add_2AddV2vae_encoder_geco/mul:z:0!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_2{
vae_encoder_geco/SignSignvae_encoder_geco/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Signy
vae_encoder_geco/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_3/y?
vae_encoder_geco/add_3AddV2vae_encoder_geco/Sign:y:0!vae_encoder_geco/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_3?
vae_encoder_geco/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/truediv_2/y?
vae_encoder_geco/truediv_2RealDivvae_encoder_geco/add_3:z:0%vae_encoder_geco/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_2?
vae_encoder_geco/IdentityIdentityvae_encoder_geco/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Identity?
vae_encoder_geco/IdentityN	IdentityNvae_encoder_geco/truediv_2:z:0!vae_encoder_geco/Reshape:output:0#vae_encoder_geco/Reshape_1:output:0vae_encoder_geco/zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-334898956*<
_output_shapes*
(:@:@:@:@2
vae_encoder_geco/IdentityN?
.vae_encoder_geco/dense_2/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp?
vae_encoder_geco/dense_2/MatMulMatMul%vae_encoder_geco/dense/Gelu/mul_1:z:06vae_encoder_geco/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_2/MatMul?
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_2/BiasAddBiasAdd)vae_encoder_geco/dense_2/MatMul:product:07vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/BiasAdd?
 vae_encoder_geco/dense_2/SigmoidSigmoid)vae_encoder_geco/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/Sigmoidv
vae_encoder_geco/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const_1?
"vae_encoder_geco/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"vae_encoder_geco/split_1/split_dim?
vae_encoder_geco/split_1Split+vae_encoder_geco/split_1/split_dim:output:0$vae_encoder_geco/dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2
vae_encoder_geco/split_1?
 vae_encoder_geco/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_2/shape?
vae_encoder_geco/Reshape_2Reshape!vae_encoder_geco/split_1:output:0)vae_encoder_geco/Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_2?
 vae_encoder_geco/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_3/shape?
vae_encoder_geco/Reshape_3Reshape!vae_encoder_geco/split_1:output:1)vae_encoder_geco/Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_3?
vae_encoder_geco/zeros_1Const*
_output_shapes
:	?*
dtype0*
valueB	?*    2
vae_encoder_geco/zeros_1?
vae_encoder_geco/mul_1Mul#vae_encoder_geco/Reshape_3:output:0!vae_encoder_geco/zeros_1:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/mul_1?
vae_encoder_geco/add_4AddV2#vae_encoder_geco/Reshape_2:output:0vae_encoder_geco/mul_1:z:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/add_4?
IdentityIdentity#vae_encoder_geco/IdentityN:output:0.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:??????????
:?????????:?????????@:??????????::::::2^
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp-vae_encoder_geco/dense/BiasAdd/ReadVariableOp2b
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp2\
,vae_encoder_geco/dense/MatMul/ReadVariableOp,vae_encoder_geco/dense/MatMul/ReadVariableOp2`
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp.vae_encoder_geco/dense/MatMul_1/ReadVariableOp2b
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp.vae_encoder_geco/dense_1/MatMul/ReadVariableOp2b
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp.vae_encoder_geco/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:??????????

#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334899796
bbn
cbn
gcn_layer_334899789
gcn_layer_334899791
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_build_adjacency_hamming_8202932
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_334899789gcn_layer_334899791*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_spectrum_conv_8203442#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:@:	?::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:D@

_output_shapes
:	?

_user_specified_namecbn
?
?
'__inference_tbh_layer_call_fn_334899078
	input_1_1
	input_1_2
	input_1_3
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_1_1	input_1_2	input_1_3input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_3348986802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:??????????
:?????????:?????????@:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:??????????

#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?	
?
F__inference_dense_9_layer_call_and_return_conditional_losses_334899879

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
4__inference_vae_encoder_geco_layer_call_fn_334899694

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:@:	?*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_3348982092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????
::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?	
?
4__inference_vae_encoder_geco_layer_call_fn_334899675

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:@:	?*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_3348981092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????
::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
??
?
$__inference__wrapped_model_334897983
	input_1_1
	input_1_2
	input_1_3
input_2
input_3=
9tbh_vae_encoder_geco_dense_matmul_readvariableop_resource>
:tbh_vae_encoder_geco_dense_biasadd_readvariableop_resource?
;tbh_vae_encoder_geco_dense_1_matmul_readvariableop_resource@
<tbh_vae_encoder_geco_dense_1_biasadd_readvariableop_resource?
;tbh_vae_encoder_geco_dense_2_matmul_readvariableop_resource@
<tbh_vae_encoder_geco_dense_2_biasadd_readvariableop_resource
identity??1tbh/vae_encoder_geco/dense/BiasAdd/ReadVariableOp?3tbh/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?0tbh/vae_encoder_geco/dense/MatMul/ReadVariableOp?2tbh/vae_encoder_geco/dense/MatMul_1/ReadVariableOp?3tbh/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?2tbh/vae_encoder_geco/dense_1/MatMul/ReadVariableOp?3tbh/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?2tbh/vae_encoder_geco/dense_2/MatMul/ReadVariableOpq
tbh/vae_encoder_geco/ShapeShape	input_1_2*
T0*
_output_shapes
:2
tbh/vae_encoder_geco/Shape?
(tbh/vae_encoder_geco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(tbh/vae_encoder_geco/strided_slice/stack?
*tbh/vae_encoder_geco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*tbh/vae_encoder_geco/strided_slice/stack_1?
*tbh/vae_encoder_geco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*tbh/vae_encoder_geco/strided_slice/stack_2?
"tbh/vae_encoder_geco/strided_sliceStridedSlice#tbh/vae_encoder_geco/Shape:output:01tbh/vae_encoder_geco/strided_slice/stack:output:03tbh/vae_encoder_geco/strided_slice/stack_1:output:03tbh/vae_encoder_geco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"tbh/vae_encoder_geco/strided_slice?
0tbh/vae_encoder_geco/dense/MatMul/ReadVariableOpReadVariableOp9tbh_vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype022
0tbh/vae_encoder_geco/dense/MatMul/ReadVariableOp?
!tbh/vae_encoder_geco/dense/MatMulMatMul	input_1_28tbh/vae_encoder_geco/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!tbh/vae_encoder_geco/dense/MatMul?
1tbh/vae_encoder_geco/dense/BiasAdd/ReadVariableOpReadVariableOp:tbh_vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1tbh/vae_encoder_geco/dense/BiasAdd/ReadVariableOp?
"tbh/vae_encoder_geco/dense/BiasAddBiasAdd+tbh/vae_encoder_geco/dense/MatMul:product:09tbh/vae_encoder_geco/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"tbh/vae_encoder_geco/dense/BiasAdd?
%tbh/vae_encoder_geco/dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%tbh/vae_encoder_geco/dense/Gelu/mul/x?
#tbh/vae_encoder_geco/dense/Gelu/mulMul.tbh/vae_encoder_geco/dense/Gelu/mul/x:output:0+tbh/vae_encoder_geco/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2%
#tbh/vae_encoder_geco/dense/Gelu/mul?
&tbh/vae_encoder_geco/dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2(
&tbh/vae_encoder_geco/dense/Gelu/Cast/x?
'tbh/vae_encoder_geco/dense/Gelu/truedivRealDiv+tbh/vae_encoder_geco/dense/BiasAdd:output:0/tbh/vae_encoder_geco/dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2)
'tbh/vae_encoder_geco/dense/Gelu/truediv?
#tbh/vae_encoder_geco/dense/Gelu/ErfErf+tbh/vae_encoder_geco/dense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2%
#tbh/vae_encoder_geco/dense/Gelu/Erf?
%tbh/vae_encoder_geco/dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%tbh/vae_encoder_geco/dense/Gelu/add/x?
#tbh/vae_encoder_geco/dense/Gelu/addAddV2.tbh/vae_encoder_geco/dense/Gelu/add/x:output:0'tbh/vae_encoder_geco/dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2%
#tbh/vae_encoder_geco/dense/Gelu/add?
%tbh/vae_encoder_geco/dense/Gelu/mul_1Mul'tbh/vae_encoder_geco/dense/Gelu/mul:z:0'tbh/vae_encoder_geco/dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2'
%tbh/vae_encoder_geco/dense/Gelu/mul_1?
2tbh/vae_encoder_geco/dense/MatMul_1/ReadVariableOpReadVariableOp9tbh_vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype024
2tbh/vae_encoder_geco/dense/MatMul_1/ReadVariableOp?
#tbh/vae_encoder_geco/dense/MatMul_1MatMul	input_1_2:tbh/vae_encoder_geco/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#tbh/vae_encoder_geco/dense/MatMul_1?
3tbh/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOpReadVariableOp:tbh_vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3tbh/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?
$tbh/vae_encoder_geco/dense/BiasAdd_1BiasAdd-tbh/vae_encoder_geco/dense/MatMul_1:product:0;tbh/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$tbh/vae_encoder_geco/dense/BiasAdd_1?
'tbh/vae_encoder_geco/dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'tbh/vae_encoder_geco/dense/Gelu_1/mul/x?
%tbh/vae_encoder_geco/dense/Gelu_1/mulMul0tbh/vae_encoder_geco/dense/Gelu_1/mul/x:output:0-tbh/vae_encoder_geco/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2'
%tbh/vae_encoder_geco/dense/Gelu_1/mul?
(tbh/vae_encoder_geco/dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2*
(tbh/vae_encoder_geco/dense/Gelu_1/Cast/x?
)tbh/vae_encoder_geco/dense/Gelu_1/truedivRealDiv-tbh/vae_encoder_geco/dense/BiasAdd_1:output:01tbh/vae_encoder_geco/dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2+
)tbh/vae_encoder_geco/dense/Gelu_1/truediv?
%tbh/vae_encoder_geco/dense/Gelu_1/ErfErf-tbh/vae_encoder_geco/dense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2'
%tbh/vae_encoder_geco/dense/Gelu_1/Erf?
'tbh/vae_encoder_geco/dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'tbh/vae_encoder_geco/dense/Gelu_1/add/x?
%tbh/vae_encoder_geco/dense/Gelu_1/addAddV20tbh/vae_encoder_geco/dense/Gelu_1/add/x:output:0)tbh/vae_encoder_geco/dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2'
%tbh/vae_encoder_geco/dense/Gelu_1/add?
'tbh/vae_encoder_geco/dense/Gelu_1/mul_1Mul)tbh/vae_encoder_geco/dense/Gelu_1/mul:z:0)tbh/vae_encoder_geco/dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2)
'tbh/vae_encoder_geco/dense/Gelu_1/mul_1?
2tbh/vae_encoder_geco/dense_1/MatMul/ReadVariableOpReadVariableOp;tbh_vae_encoder_geco_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype024
2tbh/vae_encoder_geco/dense_1/MatMul/ReadVariableOp?
#tbh/vae_encoder_geco/dense_1/MatMulMatMul+tbh/vae_encoder_geco/dense/Gelu_1/mul_1:z:0:tbh/vae_encoder_geco/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#tbh/vae_encoder_geco/dense_1/MatMul?
3tbh/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOpReadVariableOp<tbh_vae_encoder_geco_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3tbh/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?
$tbh/vae_encoder_geco/dense_1/BiasAddBiasAdd-tbh/vae_encoder_geco/dense_1/MatMul:product:0;tbh/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$tbh/vae_encoder_geco/dense_1/BiasAddz
tbh/vae_encoder_geco/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
tbh/vae_encoder_geco/Const?
$tbh/vae_encoder_geco/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$tbh/vae_encoder_geco/split/split_dim?
tbh/vae_encoder_geco/splitSplit-tbh/vae_encoder_geco/split/split_dim:output:0-tbh/vae_encoder_geco/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
tbh/vae_encoder_geco/split?
"tbh/vae_encoder_geco/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2$
"tbh/vae_encoder_geco/Reshape/shape?
tbh/vae_encoder_geco/ReshapeReshape#tbh/vae_encoder_geco/split:output:0+tbh/vae_encoder_geco/Reshape/shape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Reshape?
$tbh/vae_encoder_geco/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2&
$tbh/vae_encoder_geco/Reshape_1/shape?
tbh/vae_encoder_geco/Reshape_1Reshape#tbh/vae_encoder_geco/split:output:1-tbh/vae_encoder_geco/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2 
tbh/vae_encoder_geco/Reshape_1?
tbh/vae_encoder_geco/zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
tbh/vae_encoder_geco/zeros?
tbh/vae_encoder_geco/NegNeg%tbh/vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Neg?
tbh/vae_encoder_geco/ExpExptbh/vae_encoder_geco/Neg:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Exp?
tbh/vae_encoder_geco/Neg_1Neg%tbh/vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Neg_1?
tbh/vae_encoder_geco/Exp_1Exptbh/vae_encoder_geco/Neg_1:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Exp_1}
tbh/vae_encoder_geco/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tbh/vae_encoder_geco/add/y?
tbh/vae_encoder_geco/addAddV2tbh/vae_encoder_geco/Exp_1:y:0#tbh/vae_encoder_geco/add/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/add}
tbh/vae_encoder_geco/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tbh/vae_encoder_geco/pow/y?
tbh/vae_encoder_geco/powPowtbh/vae_encoder_geco/add:z:0#tbh/vae_encoder_geco/pow/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/pow?
tbh/vae_encoder_geco/truedivRealDivtbh/vae_encoder_geco/Exp:y:0tbh/vae_encoder_geco/pow:z:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/truediv?
tbh/vae_encoder_geco/Neg_2Neg'tbh/vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Neg_2?
tbh/vae_encoder_geco/Exp_2Exptbh/vae_encoder_geco/Neg_2:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Exp_2?
tbh/vae_encoder_geco/Neg_3Neg'tbh/vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Neg_3?
tbh/vae_encoder_geco/Exp_3Exptbh/vae_encoder_geco/Neg_3:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Exp_3?
tbh/vae_encoder_geco/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tbh/vae_encoder_geco/add_1/y?
tbh/vae_encoder_geco/add_1AddV2tbh/vae_encoder_geco/Exp_3:y:0%tbh/vae_encoder_geco/add_1/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/add_1?
tbh/vae_encoder_geco/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tbh/vae_encoder_geco/pow_1/y?
tbh/vae_encoder_geco/pow_1Powtbh/vae_encoder_geco/add_1:z:0%tbh/vae_encoder_geco/pow_1/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/pow_1?
tbh/vae_encoder_geco/truediv_1RealDivtbh/vae_encoder_geco/Exp_2:y:0tbh/vae_encoder_geco/pow_1:z:0*
T0*
_output_shapes

:@2 
tbh/vae_encoder_geco/truediv_1?
tbh/vae_encoder_geco/mulMul#tbh/vae_encoder_geco/zeros:output:0'tbh/vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/mul?
tbh/vae_encoder_geco/add_2AddV2tbh/vae_encoder_geco/mul:z:0%tbh/vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/add_2?
tbh/vae_encoder_geco/SignSigntbh/vae_encoder_geco/add_2:z:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Sign?
tbh/vae_encoder_geco/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tbh/vae_encoder_geco/add_3/y?
tbh/vae_encoder_geco/add_3AddV2tbh/vae_encoder_geco/Sign:y:0%tbh/vae_encoder_geco/add_3/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/add_3?
 tbh/vae_encoder_geco/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 tbh/vae_encoder_geco/truediv_2/y?
tbh/vae_encoder_geco/truediv_2RealDivtbh/vae_encoder_geco/add_3:z:0)tbh/vae_encoder_geco/truediv_2/y:output:0*
T0*
_output_shapes

:@2 
tbh/vae_encoder_geco/truediv_2?
tbh/vae_encoder_geco/IdentityIdentity"tbh/vae_encoder_geco/truediv_2:z:0*
T0*
_output_shapes

:@2
tbh/vae_encoder_geco/Identity?
tbh/vae_encoder_geco/IdentityN	IdentityN"tbh/vae_encoder_geco/truediv_2:z:0%tbh/vae_encoder_geco/Reshape:output:0'tbh/vae_encoder_geco/Reshape_1:output:0#tbh/vae_encoder_geco/zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-334897933*<
_output_shapes*
(:@:@:@:@2 
tbh/vae_encoder_geco/IdentityN?
2tbh/vae_encoder_geco/dense_2/MatMul/ReadVariableOpReadVariableOp;tbh_vae_encoder_geco_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype024
2tbh/vae_encoder_geco/dense_2/MatMul/ReadVariableOp?
#tbh/vae_encoder_geco/dense_2/MatMulMatMul)tbh/vae_encoder_geco/dense/Gelu/mul_1:z:0:tbh/vae_encoder_geco/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#tbh/vae_encoder_geco/dense_2/MatMul?
3tbh/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOpReadVariableOp<tbh_vae_encoder_geco_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3tbh/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?
$tbh/vae_encoder_geco/dense_2/BiasAddBiasAdd-tbh/vae_encoder_geco/dense_2/MatMul:product:0;tbh/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$tbh/vae_encoder_geco/dense_2/BiasAdd?
$tbh/vae_encoder_geco/dense_2/SigmoidSigmoid-tbh/vae_encoder_geco/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2&
$tbh/vae_encoder_geco/dense_2/Sigmoid~
tbh/vae_encoder_geco/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
tbh/vae_encoder_geco/Const_1?
&tbh/vae_encoder_geco/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&tbh/vae_encoder_geco/split_1/split_dim?
tbh/vae_encoder_geco/split_1Split/tbh/vae_encoder_geco/split_1/split_dim:output:0(tbh/vae_encoder_geco/dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2
tbh/vae_encoder_geco/split_1?
$tbh/vae_encoder_geco/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2&
$tbh/vae_encoder_geco/Reshape_2/shape?
tbh/vae_encoder_geco/Reshape_2Reshape%tbh/vae_encoder_geco/split_1:output:0-tbh/vae_encoder_geco/Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2 
tbh/vae_encoder_geco/Reshape_2?
$tbh/vae_encoder_geco/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2&
$tbh/vae_encoder_geco/Reshape_3/shape?
tbh/vae_encoder_geco/Reshape_3Reshape%tbh/vae_encoder_geco/split_1:output:1-tbh/vae_encoder_geco/Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2 
tbh/vae_encoder_geco/Reshape_3?
tbh/vae_encoder_geco/zeros_1Const*
_output_shapes
:	?*
dtype0*
valueB	?*    2
tbh/vae_encoder_geco/zeros_1?
tbh/vae_encoder_geco/mul_1Mul'tbh/vae_encoder_geco/Reshape_3:output:0%tbh/vae_encoder_geco/zeros_1:output:0*
T0*
_output_shapes
:	?2
tbh/vae_encoder_geco/mul_1?
tbh/vae_encoder_geco/add_4AddV2'tbh/vae_encoder_geco/Reshape_2:output:0tbh/vae_encoder_geco/mul_1:z:0*
T0*
_output_shapes
:	?2
tbh/vae_encoder_geco/add_4?
IdentityIdentity'tbh/vae_encoder_geco/IdentityN:output:02^tbh/vae_encoder_geco/dense/BiasAdd/ReadVariableOp4^tbh/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp1^tbh/vae_encoder_geco/dense/MatMul/ReadVariableOp3^tbh/vae_encoder_geco/dense/MatMul_1/ReadVariableOp4^tbh/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp3^tbh/vae_encoder_geco/dense_1/MatMul/ReadVariableOp4^tbh/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp3^tbh/vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:??????????
:?????????:?????????@:??????????::::::2f
1tbh/vae_encoder_geco/dense/BiasAdd/ReadVariableOp1tbh/vae_encoder_geco/dense/BiasAdd/ReadVariableOp2j
3tbh/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp3tbh/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp2d
0tbh/vae_encoder_geco/dense/MatMul/ReadVariableOp0tbh/vae_encoder_geco/dense/MatMul/ReadVariableOp2h
2tbh/vae_encoder_geco/dense/MatMul_1/ReadVariableOp2tbh/vae_encoder_geco/dense/MatMul_1/ReadVariableOp2j
3tbh/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp3tbh/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp2h
2tbh/vae_encoder_geco/dense_1/MatMul/ReadVariableOp2tbh/vae_encoder_geco/dense_1/MatMul/ReadVariableOp2j
3tbh/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp3tbh/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp2h
2tbh/vae_encoder_geco/dense_2/MatMul/ReadVariableOp2tbh/vae_encoder_geco/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:??????????

#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
A
"__inference_graph_laplacian_820340
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consth
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes

:2
ones]
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes

:2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y^
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes

:2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yS
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes

:2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Consto
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes
:2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagV
mulMuleye/diag:output:0Pow:z:0*
T0*
_output_shapes

:2
mul[
matmul_1MatMulmul:z:0	adjacency*
T0*
_output_shapes

:2

matmul_1d
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0*
_output_shapes

:2

matmul_2]
IdentityIdentitymatmul_2:product:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

::I E

_output_shapes

:
#
_user_specified_name	adjacency
?"
?
F__inference_decoder_layer_call_and_return_conditional_losses_334899758

inputs*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/BiasAddm
dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_5/Gelu/mul/x?
dense_5/Gelu/mulMuldense_5/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mulo
dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_5/Gelu/Cast/x?
dense_5/Gelu/truedivRealDivdense_5/BiasAdd:output:0dense_5/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/truedivo
dense_5/Gelu/ErfErfdense_5/Gelu/truediv:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/Erfm
dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_5/Gelu/add/x?
dense_5/Gelu/addAddV2dense_5/Gelu/add/x:output:0dense_5/Gelu/Erf:y:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/add?
dense_5/Gelu/mul_1Muldense_5/Gelu/mul:z:0dense_5/Gelu/add:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mul_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/Gelu/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?
*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/BiasAddm
dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_6/Gelu/mul/x?
dense_6/Gelu/mulMuldense_6/Gelu/mul/x:output:0dense_6/BiasAdd:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mulo
dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_6/Gelu/Cast/x?
dense_6/Gelu/truedivRealDivdense_6/BiasAdd:output:0dense_6/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/truedivo
dense_6/Gelu/ErfErfdense_6/Gelu/truediv:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/Erfm
dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_6/Gelu/add/x?
dense_6/Gelu/addAddV2dense_6/Gelu/add/x:output:0dense_6/Gelu/Erf:y:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/add?
dense_6/Gelu/mul_1Muldense_6/Gelu/mul:z:0dense_6/Gelu/add:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mul_1?
IdentityIdentitydense_6/Gelu/mul_1:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?
2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
+__inference_dense_9_layer_call_fn_334899908

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_9_layer_call_and_return_conditional_losses_3348983522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	?::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?r
?
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334899556

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddi
dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu/mul/x?
dense/Gelu/mulMuldense/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mulk
dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu/Cast/x?
dense/Gelu/truedivRealDivdense/BiasAdd:output:0dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/truedivr
dense/Gelu/ErfErfdense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/Erfi
dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu/add/x?
dense/Gelu/addAddV2dense/Gelu/add/x:output:0dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/add?
dense/Gelu/mul_1Muldense/Gelu/mul:z:0dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mul_1?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1m
dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu_1/mul/x?
dense/Gelu_1/mulMuldense/Gelu_1/mul/x:output:0dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mulo
dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu_1/Cast/x?
dense/Gelu_1/truedivRealDivdense/BiasAdd_1:output:0dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/truedivx
dense/Gelu_1/ErfErfdense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/Erfm
dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu_1/add/x?
dense/Gelu_1/addAddV2dense/Gelu_1/add/x:output:0dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/add?
dense/Gelu_1/mul_1Muldense/Gelu_1/mul:z:0dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Gelu_1/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:@2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@2
random_normalw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumrandom_normal:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:@2
clip_by_valueL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1a
mulMulclip_by_value:z:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-334899496*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Gelu/mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/SigmoidT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2	
split_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_2/shapew
	Reshape_2Reshapesplit_1:output:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_2s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_3/shapew
	Reshape_3Reshapesplit_1:output:1Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_3
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:	?*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*
_output_shapes
:	?2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:	?2
random_normal_1{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimumrandom_normal_1:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:	?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:	?2
clip_by_value_1h
mul_1MulReshape_3:output:0clip_by_value_1:z:0*
T0*
_output_shapes
:	?2
mul_1`
add_4AddV2Reshape_2:output:0	mul_1:z:0*
T0*
_output_shapes
:	?2
add_4?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identity	add_4:z:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????
::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
'__inference_tbh_layer_call_fn_334899438

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1
inputs_0_2inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_3348986802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:??????????
:?????????:?????????@:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:??????????

$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?	
?
F__inference_dense_8_layer_call_and_return_conditional_losses_334898479

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_dense_9_layer_call_fn_334899888

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_9_layer_call_and_return_conditional_losses_3348985022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334899808
bbn
cbn
gcn_layer_334899801
gcn_layer_334899803
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_build_adjacency_hamming_8202932
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_334899801gcn_layer_334899803*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_spectrum_conv_8203442#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:@:	?::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:D@

_output_shapes
:	?

_user_specified_namecbn
?
?
+__inference_decoder_layer_call_fn_334899784

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_decoder_layer_call_and_return_conditional_losses_3348984322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?
2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
A
"__inference_graph_laplacian_822105
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"?  ?  2
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consti
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes
:	?2
ones^
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes
:	?2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y_
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes
:	?2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yT
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes
:	?2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Constp
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes	
:?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0* 
_output_shapes
:
??2

eye/diagX
mulMuleye/diag:output:0Pow:z:0*
T0* 
_output_shapes
:
??2
mul]
matmul_1MatMulmul:z:0	adjacency*
T0* 
_output_shapes
:
??2

matmul_1f
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0* 
_output_shapes
:
??2

matmul_2_
IdentityIdentitymatmul_2:product:0*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??:K G
 
_output_shapes
:
??
#
_user_specified_name	adjacency
?=
?
B__inference_tbh_layer_call_and_return_conditional_losses_334898607

inputs
inputs_1
inputs_2
inputs_3
inputs_4
vae_encoder_geco_334898557
vae_encoder_geco_334898559
vae_encoder_geco_334898561
vae_encoder_geco_334898563
vae_encoder_geco_334898565
vae_encoder_geco_334898567
twin_bottleneck_334898571
twin_bottleneck_334898573
dense_8_334898576
dense_8_334898578
dense_9_334898581
dense_9_334898583
decoder_334898586
decoder_334898588
decoder_334898590
decoder_334898592
identity

identity_1

identity_2

identity_3

identity_4

identity_5??decoder/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?!dense_8/StatefulPartitionedCall_1?dense_9/StatefulPartitionedCall?!dense_9/StatefulPartitionedCall_1?'twin_bottleneck/StatefulPartitionedCall?(vae_encoder_geco/StatefulPartitionedCall?
(vae_encoder_geco/StatefulPartitionedCallStatefulPartitionedCallinputs_1vae_encoder_geco_334898557vae_encoder_geco_334898559vae_encoder_geco_334898561vae_encoder_geco_334898563vae_encoder_geco_334898565vae_encoder_geco_334898567*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:@:	?*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_3348981092*
(vae_encoder_geco/StatefulPartitionedCall?
'twin_bottleneck/StatefulPartitionedCallStatefulPartitionedCall1vae_encoder_geco/StatefulPartitionedCall:output:01vae_encoder_geco/StatefulPartitionedCall:output:1twin_bottleneck_334898571twin_bottleneck_334898573*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_3348982752)
'twin_bottleneck/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall1vae_encoder_geco/StatefulPartitionedCall:output:0dense_8_334898576dense_8_334898578*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_3348983252!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall0twin_bottleneck/StatefulPartitionedCall:output:0dense_9_334898581dense_9_334898583*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_9_layer_call_and_return_conditional_losses_3348983522!
dense_9/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall0twin_bottleneck/StatefulPartitionedCall:output:0decoder_334898586decoder_334898588decoder_334898590decoder_334898592*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_decoder_layer_call_and_return_conditional_losses_3348984002!
decoder/StatefulPartitionedCall?
!dense_8/StatefulPartitionedCall_1StatefulPartitionedCallinputs_3dense_8_334898576dense_8_334898578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_3348984792#
!dense_8/StatefulPartitionedCall_1?
!dense_9/StatefulPartitionedCall_1StatefulPartitionedCallinputs_4dense_9_334898581dense_9_334898583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_9_layer_call_and_return_conditional_losses_3348985022#
!dense_9/StatefulPartitionedCall_1?
IdentityIdentity1vae_encoder_geco/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dense_8/StatefulPartitionedCall_1 ^dense_9/StatefulPartitionedCall"^dense_9/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall)^vae_encoder_geco/StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dense_8/StatefulPartitionedCall_1 ^dense_9/StatefulPartitionedCall"^dense_9/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall)^vae_encoder_geco/StatefulPartitionedCall*
T0*
_output_shapes
:	?
2

Identity_1?

Identity_2Identity(dense_8/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dense_8/StatefulPartitionedCall_1 ^dense_9/StatefulPartitionedCall"^dense_9/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall)^vae_encoder_geco/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identity(dense_9/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dense_8/StatefulPartitionedCall_1 ^dense_9/StatefulPartitionedCall"^dense_9/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall)^vae_encoder_geco/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identity*dense_8/StatefulPartitionedCall_1:output:0 ^decoder/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dense_8/StatefulPartitionedCall_1 ^dense_9/StatefulPartitionedCall"^dense_9/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall)^vae_encoder_geco/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity*dense_9/StatefulPartitionedCall_1:output:0 ^decoder/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dense_8/StatefulPartitionedCall_1 ^dense_9/StatefulPartitionedCall"^dense_9/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall)^vae_encoder_geco/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:??????????
:?????????:?????????@:??????????::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dense_8/StatefulPartitionedCall_1!dense_8/StatefulPartitionedCall_12B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dense_9/StatefulPartitionedCall_1!dense_9/StatefulPartitionedCall_12R
'twin_bottleneck/StatefulPartitionedCall'twin_bottleneck/StatefulPartitionedCall2T
(vae_encoder_geco/StatefulPartitionedCall(vae_encoder_geco/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334898287
bbn
cbn
gcn_layer_334898280
gcn_layer_334898282
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_build_adjacency_hamming_8202932
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_334898280gcn_layer_334898282*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_spectrum_conv_8203442#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:@:	?::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:D@

_output_shapes
:	?

_user_specified_namecbn
??
?
B__inference_tbh_layer_call_and_return_conditional_losses_334899263

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_29
5vae_encoder_geco_dense_matmul_readvariableop_resource:
6vae_encoder_geco_dense_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_1_matmul_readvariableop_resource<
8vae_encoder_geco_dense_1_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_2_matmul_readvariableop_resource<
8vae_encoder_geco_dense_2_biasadd_readvariableop_resource'
#twin_bottleneck_gcn_layer_334899199'
#twin_bottleneck_gcn_layer_334899201*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource2
.decoder_dense_5_matmul_readvariableop_resource3
/decoder_dense_5_biasadd_readvariableop_resource2
.decoder_dense_6_matmul_readvariableop_resource3
/decoder_dense_6_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&decoder/dense_5/BiasAdd/ReadVariableOp?%decoder/dense_5/MatMul/ReadVariableOp?&decoder/dense_6/BiasAdd/ReadVariableOp?%decoder/dense_6/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp? dense_8/BiasAdd_1/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_8/MatMul_1/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/BiasAdd_1/ReadVariableOp?dense_9/MatMul/ReadVariableOp?dense_9/MatMul_1/ReadVariableOp?1twin_bottleneck/gcn_layer/StatefulPartitionedCall?-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?,vae_encoder_geco/dense/MatMul/ReadVariableOp?.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_2/MatMul/ReadVariableOpj
vae_encoder_geco/ShapeShape
inputs_0_1*
T0*
_output_shapes
:2
vae_encoder_geco/Shape?
$vae_encoder_geco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$vae_encoder_geco/strided_slice/stack?
&vae_encoder_geco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_1?
&vae_encoder_geco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_2?
vae_encoder_geco/strided_sliceStridedSlicevae_encoder_geco/Shape:output:0-vae_encoder_geco/strided_slice/stack:output:0/vae_encoder_geco/strided_slice/stack_1:output:0/vae_encoder_geco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
vae_encoder_geco/strided_slice?
,vae_encoder_geco/dense/MatMul/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02.
,vae_encoder_geco/dense/MatMul/ReadVariableOp?
vae_encoder_geco/dense/MatMulMatMul
inputs_0_14vae_encoder_geco/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder_geco/dense/MatMul?
-vae_encoder_geco/dense/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?
vae_encoder_geco/dense/BiasAddBiasAdd'vae_encoder_geco/dense/MatMul:product:05vae_encoder_geco/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
vae_encoder_geco/dense/BiasAdd?
!vae_encoder_geco/dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!vae_encoder_geco/dense/Gelu/mul/x?
vae_encoder_geco/dense/Gelu/mulMul*vae_encoder_geco/dense/Gelu/mul/x:output:0'vae_encoder_geco/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/mul?
"vae_encoder_geco/dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2$
"vae_encoder_geco/dense/Gelu/Cast/x?
#vae_encoder_geco/dense/Gelu/truedivRealDiv'vae_encoder_geco/dense/BiasAdd:output:0+vae_encoder_geco/dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu/truediv?
vae_encoder_geco/dense/Gelu/ErfErf'vae_encoder_geco/dense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/Erf?
!vae_encoder_geco/dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!vae_encoder_geco/dense/Gelu/add/x?
vae_encoder_geco/dense/Gelu/addAddV2*vae_encoder_geco/dense/Gelu/add/x:output:0#vae_encoder_geco/dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/add?
!vae_encoder_geco/dense/Gelu/mul_1Mul#vae_encoder_geco/dense/Gelu/mul:z:0#vae_encoder_geco/dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu/mul_1?
.vae_encoder_geco/dense/MatMul_1/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype020
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?
vae_encoder_geco/dense/MatMul_1MatMul
inputs_0_16vae_encoder_geco/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/MatMul_1?
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?
 vae_encoder_geco/dense/BiasAdd_1BiasAdd)vae_encoder_geco/dense/MatMul_1:product:07vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense/BiasAdd_1?
#vae_encoder_geco/dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#vae_encoder_geco/dense/Gelu_1/mul/x?
!vae_encoder_geco/dense/Gelu_1/mulMul,vae_encoder_geco/dense/Gelu_1/mul/x:output:0)vae_encoder_geco/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/mul?
$vae_encoder_geco/dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2&
$vae_encoder_geco/dense/Gelu_1/Cast/x?
%vae_encoder_geco/dense/Gelu_1/truedivRealDiv)vae_encoder_geco/dense/BiasAdd_1:output:0-vae_encoder_geco/dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2'
%vae_encoder_geco/dense/Gelu_1/truediv?
!vae_encoder_geco/dense/Gelu_1/ErfErf)vae_encoder_geco/dense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/Erf?
#vae_encoder_geco/dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#vae_encoder_geco/dense/Gelu_1/add/x?
!vae_encoder_geco/dense/Gelu_1/addAddV2,vae_encoder_geco/dense/Gelu_1/add/x:output:0%vae_encoder_geco/dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/add?
#vae_encoder_geco/dense/Gelu_1/mul_1Mul%vae_encoder_geco/dense/Gelu_1/mul:z:0%vae_encoder_geco/dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu_1/mul_1?
.vae_encoder_geco/dense_1/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?
vae_encoder_geco/dense_1/MatMulMatMul'vae_encoder_geco/dense/Gelu_1/mul_1:z:06vae_encoder_geco/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_1/MatMul?
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_1/BiasAddBiasAdd)vae_encoder_geco/dense_1/MatMul:product:07vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_1/BiasAddr
vae_encoder_geco/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const?
 vae_encoder_geco/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 vae_encoder_geco/split/split_dim?
vae_encoder_geco/splitSplit)vae_encoder_geco/split/split_dim:output:0)vae_encoder_geco/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder_geco/split?
vae_encoder_geco/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2 
vae_encoder_geco/Reshape/shape?
vae_encoder_geco/ReshapeReshapevae_encoder_geco/split:output:0'vae_encoder_geco/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape?
 vae_encoder_geco/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2"
 vae_encoder_geco/Reshape_1/shape?
vae_encoder_geco/Reshape_1Reshapevae_encoder_geco/split:output:1)vae_encoder_geco/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape_1?
$vae_encoder_geco/random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2&
$vae_encoder_geco/random_normal/shape?
#vae_encoder_geco/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#vae_encoder_geco/random_normal/mean?
%vae_encoder_geco/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%vae_encoder_geco/random_normal/stddev?
3vae_encoder_geco/random_normal/RandomStandardNormalRandomStandardNormal-vae_encoder_geco/random_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype025
3vae_encoder_geco/random_normal/RandomStandardNormal?
"vae_encoder_geco/random_normal/mulMul<vae_encoder_geco/random_normal/RandomStandardNormal:output:0.vae_encoder_geco/random_normal/stddev:output:0*
T0*
_output_shapes

:@2$
"vae_encoder_geco/random_normal/mul?
vae_encoder_geco/random_normalAdd&vae_encoder_geco/random_normal/mul:z:0,vae_encoder_geco/random_normal/mean:output:0*
T0*
_output_shapes

:@2 
vae_encoder_geco/random_normal?
(vae_encoder_geco/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2*
(vae_encoder_geco/clip_by_value/Minimum/y?
&vae_encoder_geco/clip_by_value/MinimumMinimum"vae_encoder_geco/random_normal:z:01vae_encoder_geco/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2(
&vae_encoder_geco/clip_by_value/Minimum?
 vae_encoder_geco/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 vae_encoder_geco/clip_by_value/y?
vae_encoder_geco/clip_by_valueMaximum*vae_encoder_geco/clip_by_value/Minimum:z:0)vae_encoder_geco/clip_by_value/y:output:0*
T0*
_output_shapes

:@2 
vae_encoder_geco/clip_by_value
vae_encoder_geco/NegNeg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Negv
vae_encoder_geco/ExpExpvae_encoder_geco/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp?
vae_encoder_geco/Neg_1Neg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_1|
vae_encoder_geco/Exp_1Expvae_encoder_geco/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_1u
vae_encoder_geco/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add/y?
vae_encoder_geco/addAddV2vae_encoder_geco/Exp_1:y:0vae_encoder_geco/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/addu
vae_encoder_geco/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow/y?
vae_encoder_geco/powPowvae_encoder_geco/add:z:0vae_encoder_geco/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow?
vae_encoder_geco/truedivRealDivvae_encoder_geco/Exp:y:0vae_encoder_geco/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv?
vae_encoder_geco/Neg_2Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_2|
vae_encoder_geco/Exp_2Expvae_encoder_geco/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_2?
vae_encoder_geco/Neg_3Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_3|
vae_encoder_geco/Exp_3Expvae_encoder_geco/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_3y
vae_encoder_geco/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_1/y?
vae_encoder_geco/add_1AddV2vae_encoder_geco/Exp_3:y:0!vae_encoder_geco/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_1y
vae_encoder_geco/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow_1/y?
vae_encoder_geco/pow_1Powvae_encoder_geco/add_1:z:0!vae_encoder_geco/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow_1?
vae_encoder_geco/truediv_1RealDivvae_encoder_geco/Exp_2:y:0vae_encoder_geco/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_1?
vae_encoder_geco/mulMul"vae_encoder_geco/clip_by_value:z:0#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/mul?
vae_encoder_geco/add_2AddV2vae_encoder_geco/mul:z:0!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_2{
vae_encoder_geco/SignSignvae_encoder_geco/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Signy
vae_encoder_geco/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_3/y?
vae_encoder_geco/add_3AddV2vae_encoder_geco/Sign:y:0!vae_encoder_geco/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_3?
vae_encoder_geco/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/truediv_2/y?
vae_encoder_geco/truediv_2RealDivvae_encoder_geco/add_3:z:0%vae_encoder_geco/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_2?
vae_encoder_geco/IdentityIdentityvae_encoder_geco/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Identity?
vae_encoder_geco/IdentityN	IdentityNvae_encoder_geco/truediv_2:z:0!vae_encoder_geco/Reshape:output:0#vae_encoder_geco/Reshape_1:output:0"vae_encoder_geco/clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-334899140*<
_output_shapes*
(:@:@:@:@2
vae_encoder_geco/IdentityN?
.vae_encoder_geco/dense_2/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp?
vae_encoder_geco/dense_2/MatMulMatMul%vae_encoder_geco/dense/Gelu/mul_1:z:06vae_encoder_geco/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_2/MatMul?
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_2/BiasAddBiasAdd)vae_encoder_geco/dense_2/MatMul:product:07vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/BiasAdd?
 vae_encoder_geco/dense_2/SigmoidSigmoid)vae_encoder_geco/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/Sigmoidv
vae_encoder_geco/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const_1?
"vae_encoder_geco/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"vae_encoder_geco/split_1/split_dim?
vae_encoder_geco/split_1Split+vae_encoder_geco/split_1/split_dim:output:0$vae_encoder_geco/dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2
vae_encoder_geco/split_1?
 vae_encoder_geco/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_2/shape?
vae_encoder_geco/Reshape_2Reshape!vae_encoder_geco/split_1:output:0)vae_encoder_geco/Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_2?
 vae_encoder_geco/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_3/shape?
vae_encoder_geco/Reshape_3Reshape!vae_encoder_geco/split_1:output:1)vae_encoder_geco/Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_3?
&vae_encoder_geco/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&vae_encoder_geco/random_normal_1/shape?
%vae_encoder_geco/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%vae_encoder_geco/random_normal_1/mean?
'vae_encoder_geco/random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'vae_encoder_geco/random_normal_1/stddev?
5vae_encoder_geco/random_normal_1/RandomStandardNormalRandomStandardNormal/vae_encoder_geco/random_normal_1/shape:output:0*
T0*
_output_shapes
:	?*
dtype027
5vae_encoder_geco/random_normal_1/RandomStandardNormal?
$vae_encoder_geco/random_normal_1/mulMul>vae_encoder_geco/random_normal_1/RandomStandardNormal:output:00vae_encoder_geco/random_normal_1/stddev:output:0*
T0*
_output_shapes
:	?2&
$vae_encoder_geco/random_normal_1/mul?
 vae_encoder_geco/random_normal_1Add(vae_encoder_geco/random_normal_1/mul:z:0.vae_encoder_geco/random_normal_1/mean:output:0*
T0*
_output_shapes
:	?2"
 vae_encoder_geco/random_normal_1?
*vae_encoder_geco/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2,
*vae_encoder_geco/clip_by_value_1/Minimum/y?
(vae_encoder_geco/clip_by_value_1/MinimumMinimum$vae_encoder_geco/random_normal_1:z:03vae_encoder_geco/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:	?2*
(vae_encoder_geco/clip_by_value_1/Minimum?
"vae_encoder_geco/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"vae_encoder_geco/clip_by_value_1/y?
 vae_encoder_geco/clip_by_value_1Maximum,vae_encoder_geco/clip_by_value_1/Minimum:z:0+vae_encoder_geco/clip_by_value_1/y:output:0*
T0*
_output_shapes
:	?2"
 vae_encoder_geco/clip_by_value_1?
vae_encoder_geco/mul_1Mul#vae_encoder_geco/Reshape_3:output:0$vae_encoder_geco/clip_by_value_1:z:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/mul_1?
vae_encoder_geco/add_4AddV2#vae_encoder_geco/Reshape_2:output:0vae_encoder_geco/mul_1:z:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/add_4?
twin_bottleneck/PartitionedCallPartitionedCall#vae_encoder_geco/IdentityN:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_build_adjacency_hamming_8202932!
twin_bottleneck/PartitionedCall?
1twin_bottleneck/gcn_layer/StatefulPartitionedCallStatefulPartitionedCallvae_encoder_geco/add_4:z:0(twin_bottleneck/PartitionedCall:output:0#twin_bottleneck_gcn_layer_334899199#twin_bottleneck_gcn_layer_334899201*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_spectrum_conv_82034423
1twin_bottleneck/gcn_layer/StatefulPartitionedCall?
twin_bottleneck/SigmoidSigmoid:twin_bottleneck/gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2
twin_bottleneck/Sigmoid?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMul#vae_encoder_geco/IdentityN:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_8/BiasAddp
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_8/Sigmoid?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMultwin_bottleneck/Sigmoid:y:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_9/BiasAddp
dense_9/SigmoidSigmoiddense_9/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_9/Sigmoid?
%decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%decoder/dense_5/MatMul/ReadVariableOp?
decoder/dense_5/MatMulMatMultwin_bottleneck/Sigmoid:y:0-decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_5/MatMul?
&decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_5/BiasAdd/ReadVariableOp?
decoder/dense_5/BiasAddBiasAdd decoder/dense_5/MatMul:product:0.decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_5/BiasAdd}
decoder/dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
decoder/dense_5/Gelu/mul/x?
decoder/dense_5/Gelu/mulMul#decoder/dense_5/Gelu/mul/x:output:0 decoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/mul
decoder/dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
decoder/dense_5/Gelu/Cast/x?
decoder/dense_5/Gelu/truedivRealDiv decoder/dense_5/BiasAdd:output:0$decoder/dense_5/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/truediv?
decoder/dense_5/Gelu/ErfErf decoder/dense_5/Gelu/truediv:z:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/Erf}
decoder/dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
decoder/dense_5/Gelu/add/x?
decoder/dense_5/Gelu/addAddV2#decoder/dense_5/Gelu/add/x:output:0decoder/dense_5/Gelu/Erf:y:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/add?
decoder/dense_5/Gelu/mul_1Muldecoder/dense_5/Gelu/mul:z:0decoder/dense_5/Gelu/add:z:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/mul_1?
%decoder/dense_6/MatMul/ReadVariableOpReadVariableOp.decoder_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02'
%decoder/dense_6/MatMul/ReadVariableOp?
decoder/dense_6/MatMulMatMuldecoder/dense_5/Gelu/mul_1:z:0-decoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/MatMul?
&decoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?
*
dtype02(
&decoder/dense_6/BiasAdd/ReadVariableOp?
decoder/dense_6/BiasAddBiasAdd decoder/dense_6/MatMul:product:0.decoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/BiasAdd}
decoder/dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
decoder/dense_6/Gelu/mul/x?
decoder/dense_6/Gelu/mulMul#decoder/dense_6/Gelu/mul/x:output:0 decoder/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/mul
decoder/dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
decoder/dense_6/Gelu/Cast/x?
decoder/dense_6/Gelu/truedivRealDiv decoder/dense_6/BiasAdd:output:0$decoder/dense_6/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/truediv?
decoder/dense_6/Gelu/ErfErf decoder/dense_6/Gelu/truediv:z:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/Erf}
decoder/dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
decoder/dense_6/Gelu/add/x?
decoder/dense_6/Gelu/addAddV2#decoder/dense_6/Gelu/add/x:output:0decoder/dense_6/Gelu/Erf:y:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/add?
decoder/dense_6/Gelu/mul_1Muldecoder/dense_6/Gelu/mul:z:0decoder/dense_6/Gelu/add:z:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/mul_1?
dense_8/MatMul_1/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_8/MatMul_1/ReadVariableOp?
dense_8/MatMul_1MatMulinputs_1'dense_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul_1?
 dense_8/BiasAdd_1/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_8/BiasAdd_1/ReadVariableOp?
dense_8/BiasAdd_1BiasAdddense_8/MatMul_1:product:0(dense_8/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdd_1
dense_8/Sigmoid_1Sigmoiddense_8/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Sigmoid_1?
dense_9/MatMul_1/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_9/MatMul_1/ReadVariableOp?
dense_9/MatMul_1MatMulinputs_2'dense_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul_1?
 dense_9/BiasAdd_1/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_9/BiasAdd_1/ReadVariableOp?
dense_9/BiasAdd_1BiasAdddense_9/MatMul_1:product:0(dense_9/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAdd_1
dense_9/Sigmoid_1Sigmoiddense_9/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Sigmoid_1?
IdentityIdentity#vae_encoder_geco/IdentityN:output:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity?

Identity_1Identitydecoder/dense_6/Gelu/mul_1:z:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?
2

Identity_1?

Identity_2Identitydense_8/Sigmoid:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identitydense_9/Sigmoid:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identitydense_8/Sigmoid_1:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_9/Sigmoid_1:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:??????????
:?????????:?????????@:??????????::::::::::::::::2P
&decoder/dense_5/BiasAdd/ReadVariableOp&decoder/dense_5/BiasAdd/ReadVariableOp2N
%decoder/dense_5/MatMul/ReadVariableOp%decoder/dense_5/MatMul/ReadVariableOp2P
&decoder/dense_6/BiasAdd/ReadVariableOp&decoder/dense_6/BiasAdd/ReadVariableOp2N
%decoder/dense_6/MatMul/ReadVariableOp%decoder/dense_6/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/BiasAdd_1/ReadVariableOp dense_8/BiasAdd_1/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2B
dense_8/MatMul_1/ReadVariableOpdense_8/MatMul_1/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/BiasAdd_1/ReadVariableOp dense_9/BiasAdd_1/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2B
dense_9/MatMul_1/ReadVariableOpdense_9/MatMul_1/ReadVariableOp2f
1twin_bottleneck/gcn_layer/StatefulPartitionedCall1twin_bottleneck/gcn_layer/StatefulPartitionedCall2^
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp-vae_encoder_geco/dense/BiasAdd/ReadVariableOp2b
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp2\
,vae_encoder_geco/dense/MatMul/ReadVariableOp,vae_encoder_geco/dense/MatMul/ReadVariableOp2`
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp.vae_encoder_geco/dense/MatMul_1/ReadVariableOp2b
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp.vae_encoder_geco/dense_1/MatMul/ReadVariableOp2b
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp.vae_encoder_geco/dense_2/MatMul/ReadVariableOp:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:??????????

$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
A
"__inference_graph_laplacian_822059
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consth
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes

:2
ones]
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes

:2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y^
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes

:2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yS
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes

:2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Consto
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes
:2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagV
mulMuleye/diag:output:0Pow:z:0*
T0*
_output_shapes

:2
mul[
matmul_1MatMulmul:z:0	adjacency*
T0*
_output_shapes

:2

matmul_1d
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0*
_output_shapes

:2

matmul_2]
IdentityIdentitymatmul_2:product:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

::I E

_output_shapes

:
#
_user_specified_name	adjacency
?2
?	
"__inference__traced_save_334899995
file_prefix1
-savev2_tbh_dense_8_kernel_read_readvariableop/
+savev2_tbh_dense_8_bias_read_readvariableop1
-savev2_tbh_dense_9_kernel_read_readvariableop/
+savev2_tbh_dense_9_bias_read_readvariableop@
<savev2_tbh_vae_encoder_geco_dense_kernel_read_readvariableop>
:savev2_tbh_vae_encoder_geco_dense_bias_read_readvariableopB
>savev2_tbh_vae_encoder_geco_dense_1_kernel_read_readvariableop@
<savev2_tbh_vae_encoder_geco_dense_1_bias_read_readvariableopB
>savev2_tbh_vae_encoder_geco_dense_2_kernel_read_readvariableop@
<savev2_tbh_vae_encoder_geco_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop9
5savev2_tbh_decoder_dense_5_kernel_read_readvariableop7
3savev2_tbh_decoder_dense_5_bias_read_readvariableop9
5savev2_tbh_decoder_dense_6_kernel_read_readvariableop7
3savev2_tbh_decoder_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'dis_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_1/bias/.ATTRIBUTES/VARIABLE_VALUEB'dis_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_tbh_dense_8_kernel_read_readvariableop+savev2_tbh_dense_8_bias_read_readvariableop-savev2_tbh_dense_9_kernel_read_readvariableop+savev2_tbh_dense_9_bias_read_readvariableop<savev2_tbh_vae_encoder_geco_dense_kernel_read_readvariableop:savev2_tbh_vae_encoder_geco_dense_bias_read_readvariableop>savev2_tbh_vae_encoder_geco_dense_1_kernel_read_readvariableop<savev2_tbh_vae_encoder_geco_dense_1_bias_read_readvariableop>savev2_tbh_vae_encoder_geco_dense_2_kernel_read_readvariableop<savev2_tbh_vae_encoder_geco_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop5savev2_tbh_decoder_dense_5_kernel_read_readvariableop3savev2_tbh_decoder_dense_5_bias_read_readvariableop5savev2_tbh_decoder_dense_6_kernel_read_readvariableop3savev2_tbh_decoder_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :@::	?::
?
?:?:
??:?:
??:?:
?
?:?:
??
:?
:
??:?:
??
:?
:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
?
?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
?
?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??
:!

_output_shapes	
:?
:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??
:!

_output_shapes	
:?
:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?	
?
F__inference_dense_8_layer_call_and_return_conditional_losses_334899859

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_dense_8_layer_call_fn_334899868

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_3348984792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334898275
bbn
cbn
gcn_layer_334898268
gcn_layer_334898270
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_build_adjacency_hamming_8202932
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_334898268gcn_layer_334898270*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_spectrum_conv_8203442#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:@:	?::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:D@

_output_shapes
:	?

_user_specified_namecbn
?
?
'__inference_signature_wrapper_334898718
	input_1_1
	input_1_2
	input_1_3
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_1_1	input_1_2	input_1_3input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__wrapped_model_3348979832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:??????????
:?????????:?????????@:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:??????????

#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
A
"__inference_graph_laplacian_822022
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"?  ?  2
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consti
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes
:	?2
ones^
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes
:	?2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y_
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes
:	?2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yT
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes
:	?2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Constp
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes	
:?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0* 
_output_shapes
:
??2

eye/diagX
mulMuleye/diag:output:0Pow:z:0*
T0* 
_output_shapes
:
??2
mul]
matmul_1MatMulmul:z:0	adjacency*
T0* 
_output_shapes
:
??2

matmul_1f
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0* 
_output_shapes
:
??2

matmul_2_
IdentityIdentitymatmul_2:product:0*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??:K G
 
_output_shapes
:
??
#
_user_specified_name	adjacency
?
?
+__inference_dense_8_layer_call_fn_334899848

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_3348983252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:@
 
_user_specified_nameinputs
?V
?
%__inference__traced_restore_334900065
file_prefix'
#assignvariableop_tbh_dense_8_kernel'
#assignvariableop_1_tbh_dense_8_bias)
%assignvariableop_2_tbh_dense_9_kernel'
#assignvariableop_3_tbh_dense_9_bias8
4assignvariableop_4_tbh_vae_encoder_geco_dense_kernel6
2assignvariableop_5_tbh_vae_encoder_geco_dense_bias:
6assignvariableop_6_tbh_vae_encoder_geco_dense_1_kernel8
4assignvariableop_7_tbh_vae_encoder_geco_dense_1_bias:
6assignvariableop_8_tbh_vae_encoder_geco_dense_2_kernel8
4assignvariableop_9_tbh_vae_encoder_geco_dense_2_bias&
"assignvariableop_10_dense_3_kernel$
 assignvariableop_11_dense_3_bias&
"assignvariableop_12_dense_4_kernel$
 assignvariableop_13_dense_4_bias2
.assignvariableop_14_tbh_decoder_dense_5_kernel0
,assignvariableop_15_tbh_decoder_dense_5_bias2
.assignvariableop_16_tbh_decoder_dense_6_kernel0
,assignvariableop_17_tbh_decoder_dense_6_bias&
"assignvariableop_18_dense_7_kernel$
 assignvariableop_19_dense_7_bias
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'dis_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_1/bias/.ATTRIBUTES/VARIABLE_VALUEB'dis_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp#assignvariableop_tbh_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_tbh_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp%assignvariableop_2_tbh_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_tbh_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_tbh_vae_encoder_geco_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp2assignvariableop_5_tbh_vae_encoder_geco_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_tbh_vae_encoder_geco_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp4assignvariableop_7_tbh_vae_encoder_geco_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_tbh_vae_encoder_geco_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp4assignvariableop_9_tbh_vae_encoder_geco_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_tbh_decoder_dense_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_tbh_decoder_dense_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp.assignvariableop_16_tbh_decoder_dense_6_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_tbh_decoder_dense_6_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20?
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
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
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
?^
?
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334899656

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
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
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddi
dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu/mul/x?
dense/Gelu/mulMuldense/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mulk
dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu/Cast/x?
dense/Gelu/truedivRealDivdense/BiasAdd:output:0dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/truedivr
dense/Gelu/ErfErfdense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/Erfi
dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu/add/x?
dense/Gelu/addAddV2dense/Gelu/add/x:output:0dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/add?
dense/Gelu/mul_1Muldense/Gelu/mul:z:0dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu/mul_1?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1m
dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/Gelu_1/mul/x?
dense/Gelu_1/mulMuldense/Gelu_1/mul/x:output:0dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mulo
dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense/Gelu_1/Cast/x?
dense/Gelu_1/truedivRealDivdense/BiasAdd_1:output:0dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/truedivx
dense/Gelu_1/ErfErfdense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/Erfm
dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense/Gelu_1/add/x?
dense/Gelu_1/addAddV2dense/Gelu_1/add/x:output:0dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/add?
dense/Gelu_1/mul_1Muldense/Gelu_1/mul:z:0dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2
dense/Gelu_1/mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Gelu_1/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1c
zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
zerosL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1^
mulMulzeros:output:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-334899605*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Gelu/mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/SigmoidT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2	
split_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_2/shapew
	Reshape_2Reshapesplit_1:output:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_2s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_3/shapew
	Reshape_3Reshapesplit_1:output:1Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_3i
zeros_1Const*
_output_shapes
:	?*
dtype0*
valueB	?*    2	
zeros_1e
mul_1MulReshape_3:output:0zeros_1:output:0*
T0*
_output_shapes
:	?2
mul_1`
add_4AddV2Reshape_2:output:0	mul_1:z:0*
T0*
_output_shapes
:	?2
add_4?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identity	add_4:z:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????
::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
'__inference_tbh_layer_call_fn_334899057
	input_1_1
	input_1_2
	input_1_3
input_2
input_3
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_1_1	input_1_2	input_1_3input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14* 
Tin
2*
Tout

2*
_collective_manager_ids
 *c
_output_shapesQ
O:@:	?
:::?????????:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_3348986072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	?
2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:??????????
:?????????:?????????@:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:??????????

#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
 __inference_spectrum_conv_820344

values
	adjacency*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulvalues%dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_7/BiasAdd?
PartitionedCallPartitionedCall	adjacency*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_graph_laplacian_8203402
PartitionedCallx
matmulMatMulPartitionedCall:output:0dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	?2
matmul?
IdentityIdentitymatmul:product:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:	?:::2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_namevalues:IE

_output_shapes

:
#
_user_specified_name	adjacency
??
?
B__inference_tbh_layer_call_and_return_conditional_losses_334898903
	input_1_1
	input_1_2
	input_1_3
input_2
input_39
5vae_encoder_geco_dense_matmul_readvariableop_resource:
6vae_encoder_geco_dense_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_1_matmul_readvariableop_resource<
8vae_encoder_geco_dense_1_biasadd_readvariableop_resource;
7vae_encoder_geco_dense_2_matmul_readvariableop_resource<
8vae_encoder_geco_dense_2_biasadd_readvariableop_resource'
#twin_bottleneck_gcn_layer_334898839'
#twin_bottleneck_gcn_layer_334898841*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource2
.decoder_dense_5_matmul_readvariableop_resource3
/decoder_dense_5_biasadd_readvariableop_resource2
.decoder_dense_6_matmul_readvariableop_resource3
/decoder_dense_6_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&decoder/dense_5/BiasAdd/ReadVariableOp?%decoder/dense_5/MatMul/ReadVariableOp?&decoder/dense_6/BiasAdd/ReadVariableOp?%decoder/dense_6/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp? dense_8/BiasAdd_1/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_8/MatMul_1/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/BiasAdd_1/ReadVariableOp?dense_9/MatMul/ReadVariableOp?dense_9/MatMul_1/ReadVariableOp?1twin_bottleneck/gcn_layer/StatefulPartitionedCall?-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?,vae_encoder_geco/dense/MatMul/ReadVariableOp?.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?.vae_encoder_geco/dense_2/MatMul/ReadVariableOpi
vae_encoder_geco/ShapeShape	input_1_2*
T0*
_output_shapes
:2
vae_encoder_geco/Shape?
$vae_encoder_geco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$vae_encoder_geco/strided_slice/stack?
&vae_encoder_geco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_1?
&vae_encoder_geco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&vae_encoder_geco/strided_slice/stack_2?
vae_encoder_geco/strided_sliceStridedSlicevae_encoder_geco/Shape:output:0-vae_encoder_geco/strided_slice/stack:output:0/vae_encoder_geco/strided_slice/stack_1:output:0/vae_encoder_geco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
vae_encoder_geco/strided_slice?
,vae_encoder_geco/dense/MatMul/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02.
,vae_encoder_geco/dense/MatMul/ReadVariableOp?
vae_encoder_geco/dense/MatMulMatMul	input_1_24vae_encoder_geco/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder_geco/dense/MatMul?
-vae_encoder_geco/dense/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp?
vae_encoder_geco/dense/BiasAddBiasAdd'vae_encoder_geco/dense/MatMul:product:05vae_encoder_geco/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
vae_encoder_geco/dense/BiasAdd?
!vae_encoder_geco/dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!vae_encoder_geco/dense/Gelu/mul/x?
vae_encoder_geco/dense/Gelu/mulMul*vae_encoder_geco/dense/Gelu/mul/x:output:0'vae_encoder_geco/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/mul?
"vae_encoder_geco/dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2$
"vae_encoder_geco/dense/Gelu/Cast/x?
#vae_encoder_geco/dense/Gelu/truedivRealDiv'vae_encoder_geco/dense/BiasAdd:output:0+vae_encoder_geco/dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu/truediv?
vae_encoder_geco/dense/Gelu/ErfErf'vae_encoder_geco/dense/Gelu/truediv:z:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/Erf?
!vae_encoder_geco/dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!vae_encoder_geco/dense/Gelu/add/x?
vae_encoder_geco/dense/Gelu/addAddV2*vae_encoder_geco/dense/Gelu/add/x:output:0#vae_encoder_geco/dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/Gelu/add?
!vae_encoder_geco/dense/Gelu/mul_1Mul#vae_encoder_geco/dense/Gelu/mul:z:0#vae_encoder_geco/dense/Gelu/add:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu/mul_1?
.vae_encoder_geco/dense/MatMul_1/ReadVariableOpReadVariableOp5vae_encoder_geco_dense_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype020
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp?
vae_encoder_geco/dense/MatMul_1MatMul	input_1_26vae_encoder_geco/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense/MatMul_1?
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOpReadVariableOp6vae_encoder_geco_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp?
 vae_encoder_geco/dense/BiasAdd_1BiasAdd)vae_encoder_geco/dense/MatMul_1:product:07vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense/BiasAdd_1?
#vae_encoder_geco/dense/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#vae_encoder_geco/dense/Gelu_1/mul/x?
!vae_encoder_geco/dense/Gelu_1/mulMul,vae_encoder_geco/dense/Gelu_1/mul/x:output:0)vae_encoder_geco/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/mul?
$vae_encoder_geco/dense/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2&
$vae_encoder_geco/dense/Gelu_1/Cast/x?
%vae_encoder_geco/dense/Gelu_1/truedivRealDiv)vae_encoder_geco/dense/BiasAdd_1:output:0-vae_encoder_geco/dense/Gelu_1/Cast/x:output:0*
T0*(
_output_shapes
:??????????2'
%vae_encoder_geco/dense/Gelu_1/truediv?
!vae_encoder_geco/dense/Gelu_1/ErfErf)vae_encoder_geco/dense/Gelu_1/truediv:z:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/Erf?
#vae_encoder_geco/dense/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#vae_encoder_geco/dense/Gelu_1/add/x?
!vae_encoder_geco/dense/Gelu_1/addAddV2,vae_encoder_geco/dense/Gelu_1/add/x:output:0%vae_encoder_geco/dense/Gelu_1/Erf:y:0*
T0*(
_output_shapes
:??????????2#
!vae_encoder_geco/dense/Gelu_1/add?
#vae_encoder_geco/dense/Gelu_1/mul_1Mul%vae_encoder_geco/dense/Gelu_1/mul:z:0%vae_encoder_geco/dense/Gelu_1/add:z:0*
T0*(
_output_shapes
:??????????2%
#vae_encoder_geco/dense/Gelu_1/mul_1?
.vae_encoder_geco/dense_1/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp?
vae_encoder_geco/dense_1/MatMulMatMul'vae_encoder_geco/dense/Gelu_1/mul_1:z:06vae_encoder_geco/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_1/MatMul?
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_1/BiasAddBiasAdd)vae_encoder_geco/dense_1/MatMul:product:07vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_1/BiasAddr
vae_encoder_geco/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const?
 vae_encoder_geco/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 vae_encoder_geco/split/split_dim?
vae_encoder_geco/splitSplit)vae_encoder_geco/split/split_dim:output:0)vae_encoder_geco/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder_geco/split?
vae_encoder_geco/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2 
vae_encoder_geco/Reshape/shape?
vae_encoder_geco/ReshapeReshapevae_encoder_geco/split:output:0'vae_encoder_geco/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape?
 vae_encoder_geco/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2"
 vae_encoder_geco/Reshape_1/shape?
vae_encoder_geco/Reshape_1Reshapevae_encoder_geco/split:output:1)vae_encoder_geco/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Reshape_1?
$vae_encoder_geco/random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2&
$vae_encoder_geco/random_normal/shape?
#vae_encoder_geco/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#vae_encoder_geco/random_normal/mean?
%vae_encoder_geco/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%vae_encoder_geco/random_normal/stddev?
3vae_encoder_geco/random_normal/RandomStandardNormalRandomStandardNormal-vae_encoder_geco/random_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype025
3vae_encoder_geco/random_normal/RandomStandardNormal?
"vae_encoder_geco/random_normal/mulMul<vae_encoder_geco/random_normal/RandomStandardNormal:output:0.vae_encoder_geco/random_normal/stddev:output:0*
T0*
_output_shapes

:@2$
"vae_encoder_geco/random_normal/mul?
vae_encoder_geco/random_normalAdd&vae_encoder_geco/random_normal/mul:z:0,vae_encoder_geco/random_normal/mean:output:0*
T0*
_output_shapes

:@2 
vae_encoder_geco/random_normal?
(vae_encoder_geco/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2*
(vae_encoder_geco/clip_by_value/Minimum/y?
&vae_encoder_geco/clip_by_value/MinimumMinimum"vae_encoder_geco/random_normal:z:01vae_encoder_geco/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2(
&vae_encoder_geco/clip_by_value/Minimum?
 vae_encoder_geco/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 vae_encoder_geco/clip_by_value/y?
vae_encoder_geco/clip_by_valueMaximum*vae_encoder_geco/clip_by_value/Minimum:z:0)vae_encoder_geco/clip_by_value/y:output:0*
T0*
_output_shapes

:@2 
vae_encoder_geco/clip_by_value
vae_encoder_geco/NegNeg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Negv
vae_encoder_geco/ExpExpvae_encoder_geco/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp?
vae_encoder_geco/Neg_1Neg!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_1|
vae_encoder_geco/Exp_1Expvae_encoder_geco/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_1u
vae_encoder_geco/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add/y?
vae_encoder_geco/addAddV2vae_encoder_geco/Exp_1:y:0vae_encoder_geco/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/addu
vae_encoder_geco/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow/y?
vae_encoder_geco/powPowvae_encoder_geco/add:z:0vae_encoder_geco/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow?
vae_encoder_geco/truedivRealDivvae_encoder_geco/Exp:y:0vae_encoder_geco/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv?
vae_encoder_geco/Neg_2Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_2|
vae_encoder_geco/Exp_2Expvae_encoder_geco/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_2?
vae_encoder_geco/Neg_3Neg#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Neg_3|
vae_encoder_geco/Exp_3Expvae_encoder_geco/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Exp_3y
vae_encoder_geco/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_1/y?
vae_encoder_geco/add_1AddV2vae_encoder_geco/Exp_3:y:0!vae_encoder_geco/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_1y
vae_encoder_geco/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/pow_1/y?
vae_encoder_geco/pow_1Powvae_encoder_geco/add_1:z:0!vae_encoder_geco/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/pow_1?
vae_encoder_geco/truediv_1RealDivvae_encoder_geco/Exp_2:y:0vae_encoder_geco/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_1?
vae_encoder_geco/mulMul"vae_encoder_geco/clip_by_value:z:0#vae_encoder_geco/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/mul?
vae_encoder_geco/add_2AddV2vae_encoder_geco/mul:z:0!vae_encoder_geco/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_2{
vae_encoder_geco/SignSignvae_encoder_geco/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Signy
vae_encoder_geco/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder_geco/add_3/y?
vae_encoder_geco/add_3AddV2vae_encoder_geco/Sign:y:0!vae_encoder_geco/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/add_3?
vae_encoder_geco/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder_geco/truediv_2/y?
vae_encoder_geco/truediv_2RealDivvae_encoder_geco/add_3:z:0%vae_encoder_geco/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder_geco/truediv_2?
vae_encoder_geco/IdentityIdentityvae_encoder_geco/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder_geco/Identity?
vae_encoder_geco/IdentityN	IdentityNvae_encoder_geco/truediv_2:z:0!vae_encoder_geco/Reshape:output:0#vae_encoder_geco/Reshape_1:output:0"vae_encoder_geco/clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-334898780*<
_output_shapes*
(:@:@:@:@2
vae_encoder_geco/IdentityN?
.vae_encoder_geco/dense_2/MatMul/ReadVariableOpReadVariableOp7vae_encoder_geco_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp?
vae_encoder_geco/dense_2/MatMulMatMul%vae_encoder_geco/dense/Gelu/mul_1:z:06vae_encoder_geco/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
vae_encoder_geco/dense_2/MatMul?
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOpReadVariableOp8vae_encoder_geco_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp?
 vae_encoder_geco/dense_2/BiasAddBiasAdd)vae_encoder_geco/dense_2/MatMul:product:07vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/BiasAdd?
 vae_encoder_geco/dense_2/SigmoidSigmoid)vae_encoder_geco/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 vae_encoder_geco/dense_2/Sigmoidv
vae_encoder_geco/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder_geco/Const_1?
"vae_encoder_geco/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"vae_encoder_geco/split_1/split_dim?
vae_encoder_geco/split_1Split+vae_encoder_geco/split_1/split_dim:output:0$vae_encoder_geco/dense_2/Sigmoid:y:0*
T0*<
_output_shapes*
(:??????????:??????????*
	num_split2
vae_encoder_geco/split_1?
 vae_encoder_geco/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_2/shape?
vae_encoder_geco/Reshape_2Reshape!vae_encoder_geco/split_1:output:0)vae_encoder_geco/Reshape_2/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_2?
 vae_encoder_geco/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 vae_encoder_geco/Reshape_3/shape?
vae_encoder_geco/Reshape_3Reshape!vae_encoder_geco/split_1:output:1)vae_encoder_geco/Reshape_3/shape:output:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/Reshape_3?
&vae_encoder_geco/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&vae_encoder_geco/random_normal_1/shape?
%vae_encoder_geco/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%vae_encoder_geco/random_normal_1/mean?
'vae_encoder_geco/random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'vae_encoder_geco/random_normal_1/stddev?
5vae_encoder_geco/random_normal_1/RandomStandardNormalRandomStandardNormal/vae_encoder_geco/random_normal_1/shape:output:0*
T0*
_output_shapes
:	?*
dtype027
5vae_encoder_geco/random_normal_1/RandomStandardNormal?
$vae_encoder_geco/random_normal_1/mulMul>vae_encoder_geco/random_normal_1/RandomStandardNormal:output:00vae_encoder_geco/random_normal_1/stddev:output:0*
T0*
_output_shapes
:	?2&
$vae_encoder_geco/random_normal_1/mul?
 vae_encoder_geco/random_normal_1Add(vae_encoder_geco/random_normal_1/mul:z:0.vae_encoder_geco/random_normal_1/mean:output:0*
T0*
_output_shapes
:	?2"
 vae_encoder_geco/random_normal_1?
*vae_encoder_geco/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2,
*vae_encoder_geco/clip_by_value_1/Minimum/y?
(vae_encoder_geco/clip_by_value_1/MinimumMinimum$vae_encoder_geco/random_normal_1:z:03vae_encoder_geco/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:	?2*
(vae_encoder_geco/clip_by_value_1/Minimum?
"vae_encoder_geco/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"vae_encoder_geco/clip_by_value_1/y?
 vae_encoder_geco/clip_by_value_1Maximum,vae_encoder_geco/clip_by_value_1/Minimum:z:0+vae_encoder_geco/clip_by_value_1/y:output:0*
T0*
_output_shapes
:	?2"
 vae_encoder_geco/clip_by_value_1?
vae_encoder_geco/mul_1Mul#vae_encoder_geco/Reshape_3:output:0$vae_encoder_geco/clip_by_value_1:z:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/mul_1?
vae_encoder_geco/add_4AddV2#vae_encoder_geco/Reshape_2:output:0vae_encoder_geco/mul_1:z:0*
T0*
_output_shapes
:	?2
vae_encoder_geco/add_4?
twin_bottleneck/PartitionedCallPartitionedCall#vae_encoder_geco/IdentityN:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_build_adjacency_hamming_8202932!
twin_bottleneck/PartitionedCall?
1twin_bottleneck/gcn_layer/StatefulPartitionedCallStatefulPartitionedCallvae_encoder_geco/add_4:z:0(twin_bottleneck/PartitionedCall:output:0#twin_bottleneck_gcn_layer_334898839#twin_bottleneck_gcn_layer_334898841*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_spectrum_conv_82034423
1twin_bottleneck/gcn_layer/StatefulPartitionedCall?
twin_bottleneck/SigmoidSigmoid:twin_bottleneck/gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2
twin_bottleneck/Sigmoid?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMul#vae_encoder_geco/IdentityN:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_8/BiasAddp
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_8/Sigmoid?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMultwin_bottleneck/Sigmoid:y:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_9/BiasAddp
dense_9/SigmoidSigmoiddense_9/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_9/Sigmoid?
%decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%decoder/dense_5/MatMul/ReadVariableOp?
decoder/dense_5/MatMulMatMultwin_bottleneck/Sigmoid:y:0-decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_5/MatMul?
&decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_5/BiasAdd/ReadVariableOp?
decoder/dense_5/BiasAddBiasAdd decoder/dense_5/MatMul:product:0.decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_5/BiasAdd}
decoder/dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
decoder/dense_5/Gelu/mul/x?
decoder/dense_5/Gelu/mulMul#decoder/dense_5/Gelu/mul/x:output:0 decoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/mul
decoder/dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
decoder/dense_5/Gelu/Cast/x?
decoder/dense_5/Gelu/truedivRealDiv decoder/dense_5/BiasAdd:output:0$decoder/dense_5/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/truediv?
decoder/dense_5/Gelu/ErfErf decoder/dense_5/Gelu/truediv:z:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/Erf}
decoder/dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
decoder/dense_5/Gelu/add/x?
decoder/dense_5/Gelu/addAddV2#decoder/dense_5/Gelu/add/x:output:0decoder/dense_5/Gelu/Erf:y:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/add?
decoder/dense_5/Gelu/mul_1Muldecoder/dense_5/Gelu/mul:z:0decoder/dense_5/Gelu/add:z:0*
T0*
_output_shapes
:	?2
decoder/dense_5/Gelu/mul_1?
%decoder/dense_6/MatMul/ReadVariableOpReadVariableOp.decoder_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02'
%decoder/dense_6/MatMul/ReadVariableOp?
decoder/dense_6/MatMulMatMuldecoder/dense_5/Gelu/mul_1:z:0-decoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/MatMul?
&decoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?
*
dtype02(
&decoder/dense_6/BiasAdd/ReadVariableOp?
decoder/dense_6/BiasAddBiasAdd decoder/dense_6/MatMul:product:0.decoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/BiasAdd}
decoder/dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
decoder/dense_6/Gelu/mul/x?
decoder/dense_6/Gelu/mulMul#decoder/dense_6/Gelu/mul/x:output:0 decoder/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/mul
decoder/dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
decoder/dense_6/Gelu/Cast/x?
decoder/dense_6/Gelu/truedivRealDiv decoder/dense_6/BiasAdd:output:0$decoder/dense_6/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/truediv?
decoder/dense_6/Gelu/ErfErf decoder/dense_6/Gelu/truediv:z:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/Erf}
decoder/dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
decoder/dense_6/Gelu/add/x?
decoder/dense_6/Gelu/addAddV2#decoder/dense_6/Gelu/add/x:output:0decoder/dense_6/Gelu/Erf:y:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/add?
decoder/dense_6/Gelu/mul_1Muldecoder/dense_6/Gelu/mul:z:0decoder/dense_6/Gelu/add:z:0*
T0*
_output_shapes
:	?
2
decoder/dense_6/Gelu/mul_1?
dense_8/MatMul_1/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_8/MatMul_1/ReadVariableOp?
dense_8/MatMul_1MatMulinput_2'dense_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul_1?
 dense_8/BiasAdd_1/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_8/BiasAdd_1/ReadVariableOp?
dense_8/BiasAdd_1BiasAdddense_8/MatMul_1:product:0(dense_8/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdd_1
dense_8/Sigmoid_1Sigmoiddense_8/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Sigmoid_1?
dense_9/MatMul_1/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_9/MatMul_1/ReadVariableOp?
dense_9/MatMul_1MatMulinput_3'dense_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul_1?
 dense_9/BiasAdd_1/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_9/BiasAdd_1/ReadVariableOp?
dense_9/BiasAdd_1BiasAdddense_9/MatMul_1:product:0(dense_9/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAdd_1
dense_9/Sigmoid_1Sigmoiddense_9/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Sigmoid_1?
IdentityIdentity#vae_encoder_geco/IdentityN:output:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity?

Identity_1Identitydecoder/dense_6/Gelu/mul_1:z:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?
2

Identity_1?

Identity_2Identitydense_8/Sigmoid:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identitydense_9/Sigmoid:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identitydense_8/Sigmoid_1:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_9/Sigmoid_1:y:0'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/BiasAdd_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp ^dense_8/MatMul_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/BiasAdd_1/ReadVariableOp^dense_9/MatMul/ReadVariableOp ^dense_9/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall.^vae_encoder_geco/dense/BiasAdd/ReadVariableOp0^vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp-^vae_encoder_geco/dense/MatMul/ReadVariableOp/^vae_encoder_geco/dense/MatMul_1/ReadVariableOp0^vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_1/MatMul/ReadVariableOp0^vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/^vae_encoder_geco/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:??????????
:?????????:?????????@:??????????::::::::::::::::2P
&decoder/dense_5/BiasAdd/ReadVariableOp&decoder/dense_5/BiasAdd/ReadVariableOp2N
%decoder/dense_5/MatMul/ReadVariableOp%decoder/dense_5/MatMul/ReadVariableOp2P
&decoder/dense_6/BiasAdd/ReadVariableOp&decoder/dense_6/BiasAdd/ReadVariableOp2N
%decoder/dense_6/MatMul/ReadVariableOp%decoder/dense_6/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/BiasAdd_1/ReadVariableOp dense_8/BiasAdd_1/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2B
dense_8/MatMul_1/ReadVariableOpdense_8/MatMul_1/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/BiasAdd_1/ReadVariableOp dense_9/BiasAdd_1/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2B
dense_9/MatMul_1/ReadVariableOpdense_9/MatMul_1/ReadVariableOp2f
1twin_bottleneck/gcn_layer/StatefulPartitionedCall1twin_bottleneck/gcn_layer/StatefulPartitionedCall2^
-vae_encoder_geco/dense/BiasAdd/ReadVariableOp-vae_encoder_geco/dense/BiasAdd/ReadVariableOp2b
/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp/vae_encoder_geco/dense/BiasAdd_1/ReadVariableOp2\
,vae_encoder_geco/dense/MatMul/ReadVariableOp,vae_encoder_geco/dense/MatMul/ReadVariableOp2`
.vae_encoder_geco/dense/MatMul_1/ReadVariableOp.vae_encoder_geco/dense/MatMul_1/ReadVariableOp2b
/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_1/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_1/MatMul/ReadVariableOp.vae_encoder_geco/dense_1/MatMul/ReadVariableOp2b
/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp/vae_encoder_geco/dense_2/BiasAdd/ReadVariableOp2`
.vae_encoder_geco/dense_2/MatMul/ReadVariableOp.vae_encoder_geco/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:??????????

#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
 __inference_spectrum_conv_822109

values
	adjacency*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulvalues%dense_7/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_7/BiasAdd?
PartitionedCallPartitionedCall	adjacency*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_graph_laplacian_8221052
PartitionedCally
matmulMatMulPartitionedCall:output:0dense_7/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
matmul?
IdentityIdentitymatmul:product:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :
??:
??::2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:H D
 
_output_shapes
:
??
 
_user_specified_namevalues:KG
 
_output_shapes
:
??
#
_user_specified_name	adjacency
?"
?
F__inference_decoder_layer_call_and_return_conditional_losses_334898432

inputs*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/BiasAddm
dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_5/Gelu/mul/x?
dense_5/Gelu/mulMuldense_5/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mulo
dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_5/Gelu/Cast/x?
dense_5/Gelu/truedivRealDivdense_5/BiasAdd:output:0dense_5/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/truedivo
dense_5/Gelu/ErfErfdense_5/Gelu/truediv:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/Erfm
dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_5/Gelu/add/x?
dense_5/Gelu/addAddV2dense_5/Gelu/add/x:output:0dense_5/Gelu/Erf:y:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/add?
dense_5/Gelu/mul_1Muldense_5/Gelu/mul:z:0dense_5/Gelu/add:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mul_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/Gelu/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?
*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/BiasAddm
dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_6/Gelu/mul/x?
dense_6/Gelu/mulMuldense_6/Gelu/mul/x:output:0dense_6/BiasAdd:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mulo
dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_6/Gelu/Cast/x?
dense_6/Gelu/truedivRealDivdense_6/BiasAdd:output:0dense_6/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/truedivo
dense_6/Gelu/ErfErfdense_6/Gelu/truediv:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/Erfm
dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_6/Gelu/add/x?
dense_6/Gelu/addAddV2dense_6/Gelu/add/x:output:0dense_6/Gelu/Erf:y:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/add?
dense_6/Gelu/mul_1Muldense_6/Gelu/mul:z:0dense_6/Gelu/add:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mul_1?
IdentityIdentitydense_6/Gelu/mul_1:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?
2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
 __inference_spectrum_conv_822122

values
	adjacency*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulvalues%dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_7/BiasAdd?
PartitionedCallPartitionedCall	adjacency*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_graph_laplacian_8203402
PartitionedCallx
matmulMatMulPartitionedCall:output:0dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	?2
matmul?
IdentityIdentitymatmul:product:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*0
_input_shapes
:	?:::2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_namevalues:IE

_output_shapes

:
#
_user_specified_name	adjacency
?"
?
F__inference_decoder_layer_call_and_return_conditional_losses_334899726

inputs*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_5/BiasAddm
dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_5/Gelu/mul/x?
dense_5/Gelu/mulMuldense_5/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mulo
dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_5/Gelu/Cast/x?
dense_5/Gelu/truedivRealDivdense_5/BiasAdd:output:0dense_5/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/truedivo
dense_5/Gelu/ErfErfdense_5/Gelu/truediv:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/Erfm
dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_5/Gelu/add/x?
dense_5/Gelu/addAddV2dense_5/Gelu/add/x:output:0dense_5/Gelu/Erf:y:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/add?
dense_5/Gelu/mul_1Muldense_5/Gelu/mul:z:0dense_5/Gelu/add:z:0*
T0*
_output_shapes
:	?2
dense_5/Gelu/mul_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/Gelu/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?
*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
2
dense_6/BiasAddm
dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_6/Gelu/mul/x?
dense_6/Gelu/mulMuldense_6/Gelu/mul/x:output:0dense_6/BiasAdd:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mulo
dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dense_6/Gelu/Cast/x?
dense_6/Gelu/truedivRealDivdense_6/BiasAdd:output:0dense_6/Gelu/Cast/x:output:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/truedivo
dense_6/Gelu/ErfErfdense_6/Gelu/truediv:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/Erfm
dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense_6/Gelu/add/x?
dense_6/Gelu/addAddV2dense_6/Gelu/add/x:output:0dense_6/Gelu/Erf:y:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/add?
dense_6/Gelu/mul_1Muldense_6/Gelu/mul:z:0dense_6/Gelu/add:z:0*
T0*
_output_shapes
:	?
2
dense_6/Gelu/mul_1?
IdentityIdentitydense_6/Gelu/mul_1:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?
2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
	input_1_1.
serving_default_input_1_1:0?????????
@
	input_1_23
serving_default_input_1_2:0??????????

?
	input_1_32
serving_default_input_1_3:0?????????
;
input_20
serving_default_input_2:0?????????@
<
input_31
serving_default_input_3:0??????????3
output_1'
StatefulPartitionedCall:0@tensorflow/serving/predict:??
?
encoder
decoder
tbn
	dis_1
	dis_2
trainable_variables
	variables
regularization_losses
		keras_api


signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?
_tf_keras_model?{"class_name": "TBH", "name": "tbh", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "TBH"}}
?
fc_1

fc_2_1

fc_2_2
reconstruction1
reconstruction2
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "VaeEncoderGeco", "name": "vae_encoder_geco", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
fc_1
fc_2
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Decoder", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
gcn
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TwinBottleneck", "name": "twin_bottleneck", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 64]}}
?

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 512]}}
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
16
 17
%18
&19"
trackable_list_wrapper
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
16
 17
%18
&19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;layer_regularization_losses
<metrics
=non_trainable_variables
>layer_metrics
trainable_variables

?layers
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

+kernel
,bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1024, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1280}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 1280]}}
?

-kernel
.bias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 1024]}}
?

/kernel
0bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 1024]}}
?

1kernel
2bias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1024, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1280}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 1280]}}
?

3kernel
4bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1280, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 1024]}}
f
+0
,1
-2
.3
/4
05
16
27
38
49"
trackable_list_wrapper
f
+0
,1
-2
.3
/4
05
16
27
38
49"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
Wnon_trainable_variables
trainable_variables

Xlayers
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

5kernel
6bias
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1024, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 512]}}
?

7kernel
8bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1280, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 1024]}}
<
50
61
72
83"
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
?
alayer_regularization_losses
bmetrics
clayer_metrics
dnon_trainable_variables
trainable_variables

elayers
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
ffc
grs
htrainable_variables
i	variables
jregularization_losses
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?graph_laplacian
?spectrum_conv
?spectrum_conv_adapt"?
_tf_keras_layer?{"class_name": "GCNLayer", "name": "gcn_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
llayer_regularization_losses
mmetrics
nlayer_metrics
onon_trainable_variables
trainable_variables

players
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@2tbh/dense_8/kernel
:2tbh/dense_8/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
qlayer_regularization_losses
rmetrics
slayer_metrics
tnon_trainable_variables
!trainable_variables

ulayers
"	variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	?2tbh/dense_9/kernel
:2tbh/dense_9/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vlayer_regularization_losses
wmetrics
xlayer_metrics
ynon_trainable_variables
'trainable_variables

zlayers
(	variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3
?
?2!tbh/vae_encoder_geco/dense/kernel
.:,?2tbh/vae_encoder_geco/dense/bias
7:5
??2#tbh/vae_encoder_geco/dense_1/kernel
0:.?2!tbh/vae_encoder_geco/dense_1/bias
7:5
??2#tbh/vae_encoder_geco/dense_2/kernel
0:.?2!tbh/vae_encoder_geco/dense_2/bias
": 
?
?2dense_3/kernel
:?2dense_3/bias
": 
??
2dense_4/kernel
:?
2dense_4/bias
.:,
??2tbh/decoder/dense_5/kernel
':%?2tbh/decoder/dense_5/bias
.:,
??
2tbh/decoder/dense_6/kernel
':%?
2tbh/decoder/dense_6/bias
": 
??2dense_7/kernel
:?2dense_7/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
{layer_regularization_losses
|metrics
}layer_metrics
~non_trainable_variables
@trainable_variables

layers
A	variables
Bregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Dtrainable_variables
?layers
E	variables
Fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Htrainable_variables
?layers
I	variables
Jregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Ltrainable_variables
?layers
M	variables
Nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Ptrainable_variables
?layers
Q	variables
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
Ytrainable_variables
?layers
Z	variables
[regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
]trainable_variables
?layers
^	variables
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

9kernel
:bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [400, 512]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
htrainable_variables
?layers
i	variables
jregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
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
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?trainable_variables
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
f0
g1"
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
?2?
'__inference_tbh_layer_call_fn_334899438
'__inference_tbh_layer_call_fn_334899057
'__inference_tbh_layer_call_fn_334899417
'__inference_tbh_layer_call_fn_334899078?
???
FullArgSpec?
args7?4
jself
jinputs

jtraining
jmask
j
continuous
varargs
 
varkw
 
defaults?
p

 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_tbh_layer_call_and_return_conditional_losses_334899263
B__inference_tbh_layer_call_and_return_conditional_losses_334899366
B__inference_tbh_layer_call_and_return_conditional_losses_334899006
B__inference_tbh_layer_call_and_return_conditional_losses_334898903?
???
FullArgSpec?
args7?4
jself
jinputs

jtraining
jmask
j
continuous
varargs
 
varkw
 
defaults?
p

 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference__wrapped_model_334897983?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2??????????

#? 
	input_1_3?????????
!?
input_2?????????@
"?
input_3??????????
?2?
4__inference_vae_encoder_geco_layer_call_fn_334899694
4__inference_vae_encoder_geco_layer_call_fn_334899675?
???
FullArgSpec7
args/?,
jself
jinputs

jtraining
j
continuous
varargs
 
varkwjkwargs
defaults?
p
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334899556
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334899656?
???
FullArgSpec7
args/?,
jself
jinputs

jtraining
j
continuous
varargs
 
varkwjkwargs
defaults?
p
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_decoder_layer_call_fn_334899771
+__inference_decoder_layer_call_fn_334899784?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
F__inference_decoder_layer_call_and_return_conditional_losses_334899758
F__inference_decoder_layer_call_and_return_conditional_losses_334899726?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
3__inference_twin_bottleneck_layer_call_fn_334899818
3__inference_twin_bottleneck_layer_call_fn_334899828?
???
FullArgSpec-
args%?"
jself
jbbn
jcbn

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334899808
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334899796?
???
FullArgSpec-
args%?"
jself
jbbn
jcbn

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_8_layer_call_fn_334899848
+__inference_dense_8_layer_call_fn_334899868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_8_layer_call_and_return_conditional_losses_334899859
F__inference_dense_8_layer_call_and_return_conditional_losses_334899839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_9_layer_call_fn_334899888
+__inference_dense_9_layer_call_fn_334899908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_9_layer_call_and_return_conditional_losses_334899899
F__inference_dense_9_layer_call_and_return_conditional_losses_334899879?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_334898718	input_1_1	input_1_2	input_1_3input_2input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
"__inference_graph_laplacian_822059
"__inference_graph_laplacian_822022?
???
FullArgSpec
args?
j	adjacency
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
 __inference_spectrum_conv_822109
 __inference_spectrum_conv_822122?
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
$__inference__wrapped_model_334897983?+,-./0???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2??????????

#? 
	input_1_3?????????
!?
input_2?????????@
"?
input_3??????????
? "*?'
%
output_1?
output_1@?
F__inference_decoder_layer_call_and_return_conditional_losses_334899726^56787?4
?
?
inputs	?
?

trainingp"?
?
0	?

? ?
F__inference_decoder_layer_call_and_return_conditional_losses_334899758^56787?4
?
?
inputs	?
?

trainingp "?
?
0	?

? ?
+__inference_decoder_layer_call_fn_334899771Q56787?4
?
?
inputs	?
?

trainingp"?	?
?
+__inference_decoder_layer_call_fn_334899784Q56787?4
?
?
inputs	?
?

trainingp "?	?
?
F__inference_dense_8_layer_call_and_return_conditional_losses_334899839J &?#
?
?
inputs@
? "?
?
0
? ?
F__inference_dense_8_layer_call_and_return_conditional_losses_334899859\ /?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? l
+__inference_dense_8_layer_call_fn_334899848= &?#
?
?
inputs@
? "?~
+__inference_dense_8_layer_call_fn_334899868O /?,
%?"
 ?
inputs?????????@
? "???????????
F__inference_dense_9_layer_call_and_return_conditional_losses_334899879]%&0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
F__inference_dense_9_layer_call_and_return_conditional_losses_334899899K%&'?$
?
?
inputs	?
? "?
?
0
? 
+__inference_dense_9_layer_call_fn_334899888P%&0?-
&?#
!?
inputs??????????
? "??????????m
+__inference_dense_9_layer_call_fn_334899908>%&'?$
?
?
inputs	?
? "?f
"__inference_graph_laplacian_822022@+?(
!?
?
	adjacency
??
? "?
??b
"__inference_graph_laplacian_822059<)?&
?
?
	adjacency
? "??
'__inference_signature_wrapper_334898718?+,-./0???
? 
???
,
	input_1_1?
	input_1_1?????????
1
	input_1_2$?!
	input_1_2??????????

0
	input_1_3#? 
	input_1_3?????????
,
input_2!?
input_2?????????@
-
input_3"?
input_3??????????"*?'
%
output_1?
output_1@?
 __inference_spectrum_conv_822109_9:F?C
<?9
?
values
??
?
	adjacency
??
? "?
??
 __inference_spectrum_conv_822122[9:C?@
9?6
?
values	?
?
	adjacency
? "?	??
B__inference_tbh_layer_call_and_return_conditional_losses_334898903?+,-./09: %&5678???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2??????????

#? 
	input_1_3?????????
!?
input_2?????????@
"?
input_3??????????
p

 
p
? "???
???
?
0/0@
?
0/1	?

?
0/2
?
0/3
?
0/4?????????
?
0/5?????????
? ?
B__inference_tbh_layer_call_and_return_conditional_losses_334899006?+,-./0???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2??????????

#? 
	input_1_3?????????
!?
input_2?????????@
"?
input_3??????????
p 

 
p
? "?
?
0@
? ?
B__inference_tbh_layer_call_and_return_conditional_losses_334899263?+,-./09: %&5678???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1??????????

$?!

inputs/0/2?????????
"?
inputs/1?????????@
#? 
inputs/2??????????
p

 
p
? "???
???
?
0/0@
?
0/1	?

?
0/2
?
0/3
?
0/4?????????
?
0/5?????????
? ?
B__inference_tbh_layer_call_and_return_conditional_losses_334899366?+,-./0???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1??????????

$?!

inputs/0/2?????????
"?
inputs/1?????????@
#? 
inputs/2??????????
p 

 
p
? "?
?
0@
? ?
'__inference_tbh_layer_call_fn_334899057?+,-./09: %&5678???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2??????????

#? 
	input_1_3?????????
!?
input_2?????????@
"?
input_3??????????
p

 
p
? "???
?
0@
?
1	?

?
2
?
3
?
4?????????
?
5??????????
'__inference_tbh_layer_call_fn_334899078?+,-./0???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2??????????

#? 
	input_1_3?????????
!?
input_2?????????@
"?
input_3??????????
p 

 
p
? "?@?
'__inference_tbh_layer_call_fn_334899417?+,-./09: %&5678???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1??????????

$?!

inputs/0/2?????????
"?
inputs/1?????????@
#? 
inputs/2??????????
p

 
p
? "???
?
0@
?
1	?

?
2
?
3
?
4?????????
?
5??????????
'__inference_tbh_layer_call_fn_334899438?+,-./0???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1??????????

$?!

inputs/0/2?????????
"?
inputs/1?????????@
#? 
inputs/2??????????
p 

 
p
? "?@?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334899796c9:>?;
4?1
?
bbn@
?
cbn	?
p
? "?
?
0	?
? ?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_334899808c9:>?;
4?1
?
bbn@
?
cbn	?
p 
? "?
?
0	?
? ?
3__inference_twin_bottleneck_layer_call_fn_334899818V9:>?;
4?1
?
bbn@
?
cbn	?
p
? "?	??
3__inference_twin_bottleneck_layer_call_fn_334899828V9:>?;
4?1
?
bbn@
?
cbn	?
p 
? "?	??
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334899556~+,-./08?5
.?+
!?
inputs??????????

p
p
? ":?7
0?-
?
0/0@
?
0/1	?
? ?
O__inference_vae_encoder_geco_layer_call_and_return_conditional_losses_334899656~+,-./08?5
.?+
!?
inputs??????????

p 
p
? ":?7
0?-
?
0/0@
?
0/1	?
? ?
4__inference_vae_encoder_geco_layer_call_fn_334899675p+,-./08?5
.?+
!?
inputs??????????

p
p
? ",?)
?
0@
?
1	??
4__inference_vae_encoder_geco_layer_call_fn_334899694p+,-./08?5
.?+
!?
inputs??????????

p 
p
? ",?)
?
0@
?
1	?