#################################
# C++ Make
#################################
CXXFLAGS=-O2 -g -std=c++11
CXXLIBS=
CXXSRC=$(wildcard *.cpp)
CXXTARGET=$(basename $(CXXSRC))
CXXDEP=$(addsuffix .dpp,$(CXXTARGET))

#################################
# C Make
#################################
CC=gcc
CFLAGS=-O2 -g
CLIBS=
CSRC=$(wildcard *.c)
CTARGET=$(basename $(CSRC))
CDEP=$(addsuffix .d,$(CTARGET))

#################################
# haskell Make : $ make hs
#################################
HS=ghc
HSFLAGS=
HSLIBS=
HSSRC=$(wildcard *.hs)
HSTARGET=$(basename $(HSSRC))
HSOBJ=$(addsuffix .o,$(HSTARGET))
HSINT=$(addsuffix .i,$(HSTARGET))

############################
# cuda Makef:: $ make cu
############################
NCC=nvcc
NFLAGS=-Xcompiler '-O2 -g' -arch sm_50 -Xptxas=-v
NSRC=$(wildcard *.cu)
NTARGET=$(basename $(NSRC))
NDEP=$(addsuffix .dcu,$(NTARGET))



.PHONY: all
all:$(CXXDEP) $(CXXTARGET) $(CDEP) $(CTARGET) $(HSTARGET) $(NDEP) $(NTARGET)
#@rm -f $(DEP)
$(CXXDEP):%.dpp:%.cpp
	@$(CXX) -MM $^ $(CXXFLAGS) | sed -e 's/\.o//' > $@
	@printf "\t$(CXX) -o $* $^ $(CXXFLAGS) $(CXXLIBS)\n" >> $@
	$(CXX) -o $* $^ $(CXXFLAGS) $(CXXLIBS)
-include *.dpp

$(CDEP):%.d:%.c
	@$(CC) -MM $^ $(CFLAGS) | sed -e 's/\.o//' > $@
	@printf "\t$(CC) -o $* $^ $(CFLAGS) $(CLIBS)\n" >> $@
	$(CC) -o $* $^ $(CFLAGS) $(CLIBS)
-include *.d

# $(HSTARGET):%:%.hs
# 	$(HS) $(HSFLAGS) -o $@ $^
$(HSTARGET):
	$(HS) --make $@.hs

#$(NTARGET):%:%.cu
#	$(NCC) -o $@ $^ $(NFLAGS)
$(NDEP):%.dcu:%.cu
	@$(NCC) -M $^ $(NFLAGS) | sed -e 's/\.o//' > $@
	@printf "\t$(NCC) -o $* $^ $(NFLAGS) $(NLIBS)\n" >> $@
	$(NCC) -o $* $^ $(NFLAGS) $(NLIBS)
-include *.dcu


#################################
# all : force all source code compiler
#################################
.PHONY: force
force: clean $(CXXTARGET) $(CTARGET) $(NTARGET) $(HSTARGET) $(HSOBJ) $(HSINT) $(NDEP) $(NTARGET)

#################################
# clean : make clean
#################################
.PHONY: clean
clean:
	rm -f $(CXXTARGET) $(CXXDEP) $(CTARGET) $(CDEP) $(HSTARGET) $(HSOBJ) $(HSINT) $(NDEP) $(NTARGET)

#################################
# memo
#################################
# DO NOT DELETE
#all:$(TARGET)
#$(TARGET):%:%.c
#       $(CC) $(CFLAGS) -o $@ $^
#static pattern rule
#targets... : target-pattern : dep-patterns...

