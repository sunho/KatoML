BINOP_CPP(operator+, Add)
BINOP_CPP(operator-, Sub)
BINOP_CPP(operator*, Mul)
BINOP_CPP(operator/, Div)
BINOP(max, Max)
BINOP(min, Min)
BINOP(matmul, MatMul)
UNIOP(log, Log)
UNIOP(exp, Exp)
UNIOP(softmax, SoftMax)
UNIOP(log_softmax, LogSoftMax)
REDUCEOP(sum, ReduceSum)
REDUCEOP(mean, ReduceMean)

#undef BINOP_CPP
#undef BINOP
#undef UNIOP
#undef REDUCEOP