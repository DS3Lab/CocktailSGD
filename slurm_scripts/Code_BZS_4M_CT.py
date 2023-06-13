
import os
import uuid
import time

template = '''#!/bin/bash
#SBATCH --job-name=bzs_6m_lr_1e5
#SBATCH --time=999:59:00
#SBATCH --output=/var/cr01_data/logs/slurm_%j.log
#SBATCH --exclusive

nvidia-smi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/root/miniconda3/etc/profile.d/mamba.sh" ]; then
    . "/root/miniconda3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

cd /work/data/
rm -rf CocktailSGD
git clone https://github.com/DS3Lab/CocktailSGD.git
cd CocktailSGD
git checkout rp

source /root/.bashrc
conda activate cocktail

netif=enp12s0
master_ip=172.27.6.25
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=RP-7B-1T-further
export WANDB_ENTITY=asdfffjj
export WANDB_DISABLED=1

export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1

export SHOW_DATA=0

ARGS="--model-name /var/cr01_data/RedPajama-Code-1B \
--tokenizer-name /var/cr01_data/RedPajama-Code-1B \
--load-pretrained-model false \
--project-name redpajama \
--model-type flash_gptneox \
--optimizer fusedadam \
--seed 42 \
--task-name \
/var/cr05_data/tokenized_stack/abap:2.667137247022618e-05,\
/var/cr05_data/tokenized_stack/actionscript:0.0002609132767128214,\
/var/cr05_data/tokenized_stack/ada:9.391299384745947e-05,\
/var/cr05_data/tokenized_stack/agda:3.309256294471651e-05,\
/var/cr05_data/tokenized_stack/ags-script:3.716233155530898e-06,\
/var/cr05_data/tokenized_stack/alloy:3.7573419293752665e-06,\
/var/cr05_data/tokenized_stack/ampl:7.399579291986301e-08,\
/var/cr05_data/tokenized_stack/antlr:2.2252179281956583e-05,\
/var/cr05_data/tokenized_stack/apacheconf:1.5210246322416286e-07,\
/var/cr05_data/tokenized_stack/api-blueprint:8.756168828850457e-06,\
/var/cr05_data/tokenized_stack/apl:3.8601138639861874e-06,\
/var/cr05_data/tokenized_stack/applescript:5.496243062992047e-06,\
/var/cr05_data/tokenized_stack/arc:7.970991248423021e-06,\
/var/cr05_data/tokenized_stack/arduino:0.0003357065798452652,\
/var/cr05_data/tokenized_stack/asciidoc:0.0004946495430371309,\
/var/cr05_data/tokenized_stack/asp:0.00017441219478950157,\
/var/cr05_data/tokenized_stack/aspectj:3.0050513680233256e-06,\
/var/cr05_data/tokenized_stack/assembly:0.0005887804133859656,\
/var/cr05_data/tokenized_stack/ats:4.945385493477511e-06,\
/var/cr05_data/tokenized_stack/augeas:3.535354550615677e-07,\
/var/cr05_data/tokenized_stack/autohotkey:3.703900523377587e-05,\
/var/cr05_data/tokenized_stack/autoit:3.935754007859825e-05,\
/var/cr05_data/tokenized_stack/awk:1.301092692174258e-05,\
/var/cr05_data/tokenized_stack/batchfile:0.00020798573038819718,\
/var/cr05_data/tokenized_stack/befunge:3.6997896459931503e-08,\
/var/cr05_data/tokenized_stack/bison:7.646231935052511e-07,\
/var/cr05_data/tokenized_stack/bitbake:4.024137871625217e-05,\
/var/cr05_data/tokenized_stack/blitzbasic:1.9115579837631278e-06,\
/var/cr05_data/tokenized_stack/blitzmax:4.978272512553006e-06,\
/var/cr05_data/tokenized_stack/bluespec:1.1781774583795966e-05,\
/var/cr05_data/tokenized_stack/boo:3.050271019252131e-06,\
/var/cr05_data/tokenized_stack/brainfuck:1.360700414248592e-05,\
/var/cr05_data/tokenized_stack/brightscript:4.850835313635464e-06,\
/var/cr05_data/tokenized_stack/bro:1.2250414605621766e-06,\
/var/cr05_data/tokenized_stack/c:0.02302097912688442,\
/var/cr05_data/tokenized_stack/c-sharp:0.0180421557567619,\
/var/cr05_data/tokenized_stack/c2hs-haskell:2.894057678643531e-06,\
/var/cr05_data/tokenized_stack/capn-proto:1.2949263760976028e-06,\
/var/cr05_data/tokenized_stack/cartocss:5.393471128381126e-06,\
/var/cr05_data/tokenized_stack/ceylon:6.092320283735388e-06,\
/var/cr05_data/tokenized_stack/chapel:1.1510456676423135e-05,\
/var/cr05_data/tokenized_stack/chuck:1.512802877472755e-06,\
/var/cr05_data/tokenized_stack/cirru:1.2106533897166476e-05,\
/var/cr05_data/tokenized_stack/clarion:3.5846850792289195e-06,\
/var/cr05_data/tokenized_stack/clean:1.537468141779376e-06,\
/var/cr05_data/tokenized_stack/click:5.056379182857306e-07,\
/var/cr05_data/tokenized_stack/clips:2.865281536952473e-06,\
/var/cr05_data/tokenized_stack/clojure:0.0002312820725258007,\
/var/cr05_data/tokenized_stack/cmake:0.00021585806057939372,\
/var/cr05_data/tokenized_stack/cobol:5.911441678820167e-06,\
/var/cr05_data/tokenized_stack/coffeescript:0.00035721057944325423,\
/var/cr05_data/tokenized_stack/coldfusion:2.4196624284795205e-05,\
/var/cr05_data/tokenized_stack/coldfusion-cfc:2.6108182268558334e-05,\
/var/cr05_data/tokenized_stack/common-lisp:0.0004534092211164606,\
/var/cr05_data/tokenized_stack/component-pascal:1.1288469297663546e-05,\
/var/cr05_data/tokenized_stack/coq:6.577403815098935e-08,\
/var/cr05_data/tokenized_stack/cpp:0.019134070564106476,\
/var/cr05_data/tokenized_stack/creole:8.057319673496195e-07,\
/var/cr05_data/tokenized_stack/crystal:0.00011107590692748326,\
/var/cr05_data/tokenized_stack/csound:8.328637580869025e-06,\
/var/cr05_data/tokenized_stack/css:0.008530530990973487,\
/var/cr05_data/tokenized_stack/csv:0.136101530745624,\
/var/cr05_data/tokenized_stack/cucumber:9.507637214725509e-05,\
/var/cr05_data/tokenized_stack/cuda:0.00021308321834489886,\
/var/cr05_data/tokenized_stack/cycript:4.83028092671328e-06,\
/var/cr05_data/tokenized_stack/cython:0.0001080790773142288,\
/var/cr05_data/tokenized_stack/d:5.837445885900304e-07,\
/var/cr05_data/tokenized_stack/darcs-patch:1.040051978262519e-06,\
/var/cr05_data/tokenized_stack/dart:0.0015783343738580626,\
/var/cr05_data/tokenized_stack/desktop:1.1457015270425456e-05,\
/var/cr05_data/tokenized_stack/diff:0.0008917561874963447,\
/var/cr05_data/tokenized_stack/digital-command-language:2.5183234857060045e-05,\
/var/cr05_data/tokenized_stack/dm:1.4001648371391856e-05,\
/var/cr05_data/tokenized_stack/dns-zone:1.1341910703661225e-05,\
/var/cr05_data/tokenized_stack/dockerfile:0.00039767394553826604,\
/var/cr05_data/tokenized_stack/dogescript:1.315480763019787e-07,\
/var/cr05_data/tokenized_stack/dylan:1.3084922714662442e-05,\
/var/cr05_data/tokenized_stack/eagle:0.000692810276476524,\
/var/cr05_data/tokenized_stack/ec:6.770615052167466e-06,\
/var/cr05_data/tokenized_stack/ecere-projects:1.8910035968409438e-07,\
/var/cr05_data/tokenized_stack/ecl:9.977099412028197e-06,\
/var/cr05_data/tokenized_stack/edn:0.0002497070249628466,\
/var/cr05_data/tokenized_stack/eiffel:5.077344657517934e-05,\
/var/cr05_data/tokenized_stack/elixir:0.00036072537960694776,\
/var/cr05_data/tokenized_stack/elm:0.00013549451859103806,\
/var/cr05_data/tokenized_stack/emacs-lisp:0.00018972521304652876,\
/var/cr05_data/tokenized_stack/emberscript:1.2550508654685654e-05,\
/var/cr05_data/tokenized_stack/erlang:0.0002681443100320458,\
/var/cr05_data/tokenized_stack/f-sharp:0.00033590390195971815,\
/var/cr05_data/tokenized_stack/factor:1.2719054627447564e-05,\
/var/cr05_data/tokenized_stack/fancy:7.399579291986301e-07,\
/var/cr05_data/tokenized_stack/fantom:4.4685237168828384e-06,\
/var/cr05_data/tokenized_stack/fish:2.2589271227480403e-05,\
/var/cr05_data/tokenized_stack/flux:1.8745600873031963e-06,\
/var/cr05_data/tokenized_stack/forth:9.405687455591477e-06,\
/var/cr05_data/tokenized_stack/fortran:0.0006790552807481985,\
/var/cr05_data/tokenized_stack/freemarker:0.00010689925550489543,\
/var/cr05_data/tokenized_stack/g-code:0.00036226284774872714,\
/var/cr05_data/tokenized_stack/gams:2.3991080415573365e-05,\
/var/cr05_data/tokenized_stack/gap:4.99882689947519e-06,\
/var/cr05_data/tokenized_stack/gas:0.0003585466145931962,\
/var/cr05_data/tokenized_stack/gdscript:0.00014282010209010448,\
/var/cr05_data/tokenized_stack/genshi:2.6844029320372526e-06,\
/var/cr05_data/tokenized_stack/gentoo-ebuild:2.473514922215643e-05,\
/var/cr05_data/tokenized_stack/gentoo-eclass:1.3401460273264078e-06,\
/var/cr05_data/tokenized_stack/gettext-catalog:0.0016163476570319499,\
/var/cr05_data/tokenized_stack/glsl:0.0003037239537943466,\
/var/cr05_data/tokenized_stack/glyph:3.042049264483257e-07,\
/var/cr05_data/tokenized_stack/gnuplot:0.0010040119162331614,\
/var/cr05_data/tokenized_stack/go:0.01052491493227825,\
/var/cr05_data/tokenized_stack/golo:3.7820071936818876e-07,\
/var/cr05_data/tokenized_stack/gosu:1.685459727619102e-06,\
/var/cr05_data/tokenized_stack/grace:4.6041826705692543e-07,\
/var/cr05_data/tokenized_stack/grammatical-framework:1.432229680737793e-05,\
/var/cr05_data/tokenized_stack/graphql:4.855768366496788e-05,\
/var/cr05_data/tokenized_stack/graphviz_dot:0.00027953966214170473,\
/var/cr05_data/tokenized_stack/groff:0.0009618260925140704,\
/var/cr05_data/tokenized_stack/groovy:0.00041861475493458727,\
/var/cr05_data/tokenized_stack/groovy-server-pages:2.7699091816335388e-05,\
/var/cr05_data/tokenized_stack/haml:9.731268944438874e-05,\
/var/cr05_data/tokenized_stack/handlebars:0.00027466827244114706,\
/var/cr05_data/tokenized_stack/harbour:6.289642398188357e-07,\
/var/cr05_data/tokenized_stack/haskell:0.0011046503054815595,\
/var/cr05_data/tokenized_stack/haxe:0.0002022962760881366,\
/var/cr05_data/tokenized_stack/hcl:0.0003470895993227708,\
/var/cr05_data/tokenized_stack/hlsl:3.574818973506271e-05,\
/var/cr05_data/tokenized_stack/html:0.05137558734006472,\
/var/cr05_data/tokenized_stack/html_django:6.075465686459197e-05,\
/var/cr05_data/tokenized_stack/html_eex:1.2423071455768112e-05,\
/var/cr05_data/tokenized_stack/html_erb:0.0005452749980264705,\
/var/cr05_data/tokenized_stack/html_php:0.0001259038416531469,\
/var/cr05_data/tokenized_stack/http:2.3538883903285313e-05,\
/var/cr05_data/tokenized_stack/hy:4.086212120130213e-06,\
/var/cr05_data/tokenized_stack/idl:2.4130850246644217e-06,\
/var/cr05_data/tokenized_stack/idris:1.3027370431280326e-05,\
/var/cr05_data/tokenized_stack/igor-pro:4.1026556296679604e-06,\
/var/cr05_data/tokenized_stack/inform-7:5.821002376362557e-06,\
/var/cr05_data/tokenized_stack/ini:0.0014999070551177766,\
/var/cr05_data/tokenized_stack/inno-setup:6.380081700645966e-06,\
/var/cr05_data/tokenized_stack/io:3.769674561528577e-06,\
/var/cr05_data/tokenized_stack/ioke:5.878554659744673e-07,\
/var/cr05_data/tokenized_stack/irc-log:1.8992253516098173e-06,\
/var/cr05_data/tokenized_stack/isabelle:2.5754646813496766e-05,\
/var/cr05_data/tokenized_stack/j:7.20636805491777e-06,\
/var/cr05_data/tokenized_stack/jade:9.223575587460924e-05,\
/var/cr05_data/tokenized_stack/jasmin:1.2521732512994596e-05,\
/var/cr05_data/tokenized_stack/java:0.03805731889242949,\
/var/cr05_data/tokenized_stack/java-server-pages:0.0004423961806035543,\
/var/cr05_data/tokenized_stack/javascript:0.06417502606388756,\
/var/cr05_data/tokenized_stack/jflex:4.1930949321255705e-06,\
/var/cr05_data/tokenized_stack/json:0.10827008512101928,\
/var/cr05_data/tokenized_stack/json5:3.8880678302003575e-05,\
/var/cr05_data/tokenized_stack/jsoniq:2.4295285342021687e-06,\
/var/cr05_data/tokenized_stack/jsonld:3.7770741408205634e-05,\
/var/cr05_data/tokenized_stack/jsx:0.0013740319896063207,\
/var/cr05_data/tokenized_stack/julia:0.0006808846211842728,\
/var/cr05_data/tokenized_stack/jupyter-notebook:0.12109410689160105,\
/var/cr05_data/tokenized_stack/kicad:0.0008632020331840464,\
/var/cr05_data/tokenized_stack/kit:2.1582106268293377e-06,\
/var/cr05_data/tokenized_stack/kotlin:0.0030734152589265104,\
/var/cr05_data/tokenized_stack/krl:5.385249373612253e-07,\
/var/cr05_data/tokenized_stack/labview:9.311137275749429e-06,\
/var/cr05_data/tokenized_stack/lasso:2.054205429003086e-05,\
/var/cr05_data/tokenized_stack/latte:6.306085907726103e-06,\
/var/cr05_data/tokenized_stack/lean:4.395350099439863e-05,\
/var/cr05_data/tokenized_stack/less:0.0004611746684956618,\
/var/cr05_data/tokenized_stack/lex:3.381196648699296e-05,\
/var/cr05_data/tokenized_stack/lfe:9.372800436515981e-07,\
/var/cr05_data/tokenized_stack/lilypond:1.2406627946230364e-05,\
/var/cr05_data/tokenized_stack/linker-script:3.360231174038668e-05,\
/var/cr05_data/tokenized_stack/liquid:4.528542526695616e-05,\
/var/cr05_data/tokenized_stack/literate-agda:2.3637544960511797e-06,\
/var/cr05_data/tokenized_stack/literate-coffeescript:2.351421863897869e-06,\
/var/cr05_data/tokenized_stack/literate-haskell:2.6206843325784818e-05,\
/var/cr05_data/tokenized_stack/livescript:1.6846375521422146e-05,\
/var/cr05_data/tokenized_stack/llvm:0.000184504398768294,\
/var/cr05_data/tokenized_stack/logos:0.00011416317584319532,\
/var/cr05_data/tokenized_stack/logtalk:2.799507498801484e-06,\
/var/cr05_data/tokenized_stack/lolcode:6.330751172032724e-07,\
/var/cr05_data/tokenized_stack/lookml:1.3976983107085236e-06,\
/var/cr05_data/tokenized_stack/lsl:6.075876774197641e-06,\
/var/cr05_data/tokenized_stack/lua:0.001393081795405801,\
/var/cr05_data/tokenized_stack/m:8.221754768873668e-08,\
/var/cr05_data/tokenized_stack/m4:3.890123268892576e-05,\
/var/cr05_data/tokenized_stack/makefile:0.0007203695984617886,\
/var/cr05_data/tokenized_stack/mako:1.195854231132675e-05,\
/var/cr05_data/tokenized_stack/maple:8.308083193946841e-06,\
/var/cr05_data/tokenized_stack/markdown:0.04738492023210359,\
/var/cr05_data/tokenized_stack/mask:2.5076352045064687e-06,\
/var/cr05_data/tokenized_stack/mathematica:0.0007671924918705241,\
/var/cr05_data/tokenized_stack/matlab:6.149872567117504e-06,\
/var/cr05_data/tokenized_stack/max:0.0001135465442355298,\
/var/cr05_data/tokenized_stack/maxscript:1.780009907461149e-06,\
/var/cr05_data/tokenized_stack/mediawiki:5.4025150586268876e-05,\
/var/cr05_data/tokenized_stack/metal:6.030657122968835e-06,\
/var/cr05_data/tokenized_stack/mirah:8.891827782536873e-06,\
/var/cr05_data/tokenized_stack/modelica:5.283299614478219e-05,\
/var/cr05_data/tokenized_stack/module-management-system:6.947382779698249e-07,\
/var/cr05_data/tokenized_stack/monkey:3.2558148884739726e-06,\
/var/cr05_data/tokenized_stack/moonscript:7.555792632594901e-06,\
/var/cr05_data/tokenized_stack/mtml:5.220814278234779e-07,\
/var/cr05_data/tokenized_stack/muf:7.194035422764459e-07,\
/var/cr05_data/tokenized_stack/mupad:3.6093503435355405e-06,\
/var/cr05_data/tokenized_stack/myghty:5.755228338211568e-08,\
/var/cr05_data/tokenized_stack/nesc:5.7420735305813697e-05,\
/var/cr05_data/tokenized_stack/netlinx:8.139537221184931e-07,\
/var/cr05_data/tokenized_stack/netlogo:1.0507402594620548e-05,\
/var/cr05_data/tokenized_stack/nginx:4.933052861324201e-08,\
/var/cr05_data/tokenized_stack/nimrod:0.0001469967535126923,\
/var/cr05_data/tokenized_stack/ninja:9.681116240348745e-06,\
/var/cr05_data/tokenized_stack/nit:1.6032421799303654e-07,\
/var/cr05_data/tokenized_stack/nix:0.00019809495940124216,\
/var/cr05_data/tokenized_stack/nsis:9.574233428353387e-06,\
/var/cr05_data/tokenized_stack/nu:9.866105722648401e-07,\
/var/cr05_data/tokenized_stack/numpy:3.6997896459931503e-08,\
/var/cr05_data/tokenized_stack/objdump:2.3168904938685997e-05,\
/var/cr05_data/tokenized_stack/objective-cpp:0.0002500071190119105,\
/var/cr05_data/tokenized_stack/objective-j:2.248649929286948e-06,\
/var/cr05_data/tokenized_stack/ocaml:0.0004538531958739798,\
/var/cr05_data/tokenized_stack/octave:8.797277602694825e-07,\
/var/cr05_data/tokenized_stack/omgrofl:2.877614169105784e-08,\
/var/cr05_data/tokenized_stack/ooc:2.030773427911796e-06,\
/var/cr05_data/tokenized_stack/opa:1.586798670392618e-06,\
/var/cr05_data/tokenized_stack/opal:2.055438692218417e-07,\
/var/cr05_data/tokenized_stack/opencl:4.5137433681116435e-05,\
/var/cr05_data/tokenized_stack/openscad:3.660325223102557e-05,\
/var/cr05_data/tokenized_stack/org:0.00018570477496454953,\
/var/cr05_data/tokenized_stack/ox:4.3164212536586757e-07,\
/var/cr05_data/tokenized_stack/oxygene:6.166316076655251e-08,\
/var/cr05_data/tokenized_stack/oz:1.9197797385320013e-06,\
/var/cr05_data/tokenized_stack/pan:4.982383389937443e-06,\
/var/cr05_data/tokenized_stack/papyrus:2.2400170867796307e-05,\
/var/cr05_data/tokenized_stack/parrot:5.755228338211568e-08,\
/var/cr05_data/tokenized_stack/parrot-assembly:2.5898527521952053e-07,\
/var/cr05_data/tokenized_stack/parrot-internal-representation:6.762393297398592e-06,\
/var/cr05_data/tokenized_stack/pascal:0.0005944616459312573,\
/var/cr05_data/tokenized_stack/pawn:1.59707586385371e-05,\
/var/cr05_data/tokenized_stack/perl:0.0011482913797947409,\
/var/cr05_data/tokenized_stack/perl6:1.655450322712713e-05,\
/var/cr05_data/tokenized_stack/php:0.027595337382029586,\
/var/cr05_data/tokenized_stack/piglatin:1.4141418202462708e-06,\
/var/cr05_data/tokenized_stack/pike:1.8868927194565068e-06,\
/var/cr05_data/tokenized_stack/pod:5.6240913496480326e-05,\
/var/cr05_data/tokenized_stack/pogoscript:2.3020913352846272e-07,\
/var/cr05_data/tokenized_stack/pony:6.848721722471765e-06,\
/var/cr05_data/tokenized_stack/postscript:0.0007403854604466116,\
/var/cr05_data/tokenized_stack/pov-ray-sdl:5.775782725133752e-06,\
/var/cr05_data/tokenized_stack/powershell:0.0005793130627696076,\
/var/cr05_data/tokenized_stack/processing:9.203843376015628e-05,\
/var/cr05_data/tokenized_stack/prolog:3.786118071066324e-06,\
/var/cr05_data/tokenized_stack/propeller-spin:7.124150507229033e-06,\
/var/cr05_data/tokenized_stack/protocol-buffer:0.00019181353875782268,\
/var/cr05_data/tokenized_stack/pure-data:4.1548637724503084e-05,\
/var/cr05_data/tokenized_stack/purebasic:1.7886427499684664e-05,\
/var/cr05_data/tokenized_stack/purescript:5.283710702216663e-05,\
/var/cr05_data/tokenized_stack/python:0.02870092907843218,\
/var/cr05_data/tokenized_stack/python-traceback:4.5219651228805174e-08,\
/var/cr05_data/tokenized_stack/qmake:7.226922441839954e-06,\
/var/cr05_data/tokenized_stack/qml:7.42835543367736e-05,\
/var/cr05_data/tokenized_stack/r:0.0001641391122057939,\
/var/cr05_data/tokenized_stack/racket:1.6221522158987746e-05,\
/var/cr05_data/tokenized_stack/ragel-in-ruby-host:4.373973537040791e-06,\
/var/cr05_data/tokenized_stack/raml:1.34261255375707e-05,\
/var/cr05_data/tokenized_stack/rdoc:1.6912149559573134e-05,\
/var/cr05_data/tokenized_stack/realbasic:2.9557208394100836e-06,\
/var/cr05_data/tokenized_stack/rebol:2.7090681963438736e-06,\
/var/cr05_data/tokenized_stack/red:1.3701554322327968e-05,\
/var/cr05_data/tokenized_stack/redcode:1.4552505940906393e-06,\
/var/cr05_data/tokenized_stack/renderscript:7.317361744297564e-07,\
/var/cr05_data/tokenized_stack/renpy:4.307377323412915e-05,\
/var/cr05_data/tokenized_stack/restructuredtext:0.0019725757367803237,\
/var/cr05_data/tokenized_stack/rhtml:5.870332904975799e-06,\
/var/cr05_data/tokenized_stack/rmarkdown:2.7830639892637368e-05,\
/var/cr05_data/tokenized_stack/robotframework:1.8161856284441933e-05,\
/var/cr05_data/tokenized_stack/rouge:8.098428447340563e-07,\
/var/cr05_data/tokenized_stack/ruby:0.00404904567769752,\
/var/cr05_data/tokenized_stack/rust:0.0034240566563094343,\
/var/cr05_data/tokenized_stack/sage:6.799391193858523e-06,\
/var/cr05_data/tokenized_stack/saltstack:2.1471112578913583e-05,\
/var/cr05_data/tokenized_stack/sas:3.559197639445411e-05,\
/var/cr05_data/tokenized_stack/sass:0.00010145234297051663,\
/var/cr05_data/tokenized_stack/scala:0.002246294396545666,\
/var/cr05_data/tokenized_stack/scaml:7.810667030429985e-08,\
/var/cr05_data/tokenized_stack/scheme:0.00010902457911264928,\
/var/cr05_data/tokenized_stack/scilab:3.9834401855192924e-06,\
/var/cr05_data/tokenized_stack/scss:0.002396744287061285,\
/var/cr05_data/tokenized_stack/self:1.562133406085997e-07,\
/var/cr05_data/tokenized_stack/shell:0.0023040316694100814,\
/var/cr05_data/tokenized_stack/shellsession:3.6997896459931503e-08,\
/var/cr05_data/tokenized_stack/shen:4.850835313635464e-07,\
/var/cr05_data/tokenized_stack/slash:1.5329461766564956e-05,\
/var/cr05_data/tokenized_stack/slim:3.688690277055171e-05,\
/var/cr05_data/tokenized_stack/smali:0.0005594369706158554,\
/var/cr05_data/tokenized_stack/smalltalk:0.00047044469699756685,\
/var/cr05_data/tokenized_stack/smarty:0.0002099055101267292,\
/var/cr05_data/tokenized_stack/smt:5.453489938193904e-05,\
/var/cr05_data/tokenized_stack/solidity:0.0004634109857927954,\
/var/cr05_data/tokenized_stack/sourcepawn:4.913731737617348e-05,\
/var/cr05_data/tokenized_stack/sparql:1.2772496033445244e-05,\
/var/cr05_data/tokenized_stack/sqf:5.6319020166784625e-05,\
/var/cr05_data/tokenized_stack/sql:0.004314776902704901,\
/var/cr05_data/tokenized_stack/squirrel:1.520613544503185e-05,\
/var/cr05_data/tokenized_stack/stan:5.956661330048972e-06,\
/var/cr05_data/tokenized_stack/standard-ml:0.0002555526926035158,\
/var/cr05_data/tokenized_stack/stata:0.00012930353725007618,\
/var/cr05_data/tokenized_stack/ston:5.549684468989726e-07,\
/var/cr05_data/tokenized_stack/stylus:8.588856119303878e-05,\
/var/cr05_data/tokenized_stack/supercollider:7.317361744297565e-06,\
/var/cr05_data/tokenized_stack/svg:0.04019756087421145,\
/var/cr05_data/tokenized_stack/swift:0.0028074250486439093,\
/var/cr05_data/tokenized_stack/systemverilog:0.0001668605130342911,\
/var/cr05_data/tokenized_stack/tcl:0.00015085275649929407,\
/var/cr05_data/tokenized_stack/tcsh:8.106650202109437e-06,\
/var/cr05_data/tokenized_stack/tea:8.246420033180289e-06,\
/var/cr05_data/tokenized_stack/tex:0.0025704864088371233,\
/var/cr05_data/tokenized_stack/text:0.1262479138684695,\
/var/cr05_data/tokenized_stack/textile:1.951844582130609e-05,\
/var/cr05_data/tokenized_stack/thrift:6.244422746959551e-06,\
/var/cr05_data/tokenized_stack/toml:0.00036831405925861815,\
/var/cr05_data/tokenized_stack/turing:1.4634723488595128e-06,\
/var/cr05_data/tokenized_stack/turtle:0.0005347264866580056,\
/var/cr05_data/tokenized_stack/twig:0.00046709844280663526,\
/var/cr05_data/tokenized_stack/txl:7.276252970453196e-07,\
/var/cr05_data/tokenized_stack/typescript:0.014886165303814208,\
/var/cr05_data/tokenized_stack/unified-parallel-c:4.850835313635464e-07,\
/var/cr05_data/tokenized_stack/unity3d-asset:0.0016917575917720592,\
/var/cr05_data/tokenized_stack/uno:4.1930949321255705e-06,\
/var/cr05_data/tokenized_stack/unrealscript:2.6461717723619902e-05,\
/var/cr05_data/tokenized_stack/urweb:4.0985447522835235e-06,\
/var/cr05_data/tokenized_stack/vala:8.883606027767999e-06,\
/var/cr05_data/tokenized_stack/vcl:1.8704492099187595e-06,\
/var/cr05_data/tokenized_stack/verilog:1.8910035968409438e-07,\
/var/cr05_data/tokenized_stack/vhdl:0.0004470990243313501,\
/var/cr05_data/tokenized_stack/viml:0.0001584455470283489,\
/var/cr05_data/tokenized_stack/visual-basic:0.00046027849722585457,\
/var/cr05_data/tokenized_stack/volt:4.805615662406659e-06,\
/var/cr05_data/tokenized_stack/vue:0.003163303703814606,\
/var/cr05_data/tokenized_stack/web-ontology-language:0.00015389480576377733,\
/var/cr05_data/tokenized_stack/webassembly:2.14505581919914e-05,\
/var/cr05_data/tokenized_stack/webidl:1.6196856894681125e-06,\
/var/cr05_data/tokenized_stack/wisp:2.507635204506469e-07,\
/var/cr05_data/tokenized_stack/x10:6.454077493565829e-07,\
/var/cr05_data/tokenized_stack/xbase:1.7676772753078386e-05,\
/var/cr05_data/tokenized_stack/xc:1.7964534169988965e-06,\
/var/cr05_data/tokenized_stack/xml:0.03283954255258181,\
/var/cr05_data/tokenized_stack/xojo:8.044987041342885e-06,\
/var/cr05_data/tokenized_stack/xpages:5.056379182857306e-07,\
/var/cr05_data/tokenized_stack/xproc:2.2650934388246957e-06,\
/var/cr05_data/tokenized_stack/xquery:1.761922046969627e-05,\
/var/cr05_data/tokenized_stack/xs:7.329694376450875e-06,\
/var/cr05_data/tokenized_stack/xslt:0.00015093086316959837,\
/var/cr05_data/tokenized_stack/xtend:1.852772437165681e-05,\
/var/cr05_data/tokenized_stack/yacc:0.00010640595021876302,\
/var/cr05_data/tokenized_stack/yaml:0.009942896912189681,\
/var/cr05_data/tokenized_stack/yang:3.443681984942736e-05,\
/var/cr05_data/tokenized_stack/zephir:3.3544759457004566e-06,\
/var/cr05_data/tokenized_stack/zig:5.4633560439165525e-05,\
/var/cr05_data/tokenized_stack/zimpl:7.070709101231354e-07 \
--checkpoint-path /var/cr05_data/model_ckpts/$WANDB_NAME \
--num-layers {{N_LAYER_PER_DEVICE}} --embedding-dim 2048 \
--initial-loss-scale 4096 \
--total-steps 262144 --warmup-steps 1000 --train-warmup-steps 0 \
--stop-steps 262145 \
--checkpoint-steps 1000 \
--lr 2e-4 --seq-length 8192 --batch-size 16 --micro-batch-size 16 --gradient-accumulate-step 1 \
--dist-url tcp://${master_ip}:8956 \
--world-size $(({{PP_DEGREE}}*{{DP_DEGREE}})) --pipeline-group-size {{PP_DEGREE}} --data-group-size {{DP_DEGREE}} \
--job-id {{JOB_ID}} --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 1 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 2 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 3 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 5 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
    & \
wait)


'''

if __name__ == '__main__':

    job_id = str(uuid.uuid4())
    pp_degree=1
    dp_degree=32
    n_layer_per_device=16
    node_size=4

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', str(pp_degree))
    template = template.replace('{{DP_DEGREE}}', str(dp_degree))
    template = template.replace('{{N_LAYER_PER_DEVICE}}', str(n_layer_per_device))

    with open('slurm_scripts/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(node_size):
        os.system('sbatch slurm_scripts/train_to_submit.slurm.sh')
        if i == 0:
            time.sleep(6)
