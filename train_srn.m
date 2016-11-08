%
% train_srn(sents,targs,hids,numpres,lr,mom)
% . sents => distributed input corpus
% . targs => distributed output corpus for training/prediction
% . hids => num hidden units 
% . lr => learning rate
% . mom => momentum
%
% outputs [wih wch who biash biaso err cosne]
% . wih => weights from input to hidden
% . wch => " context to hidden
% . who => " hidden to output
% . biash => " bias to hidden
% . biaso => " bias to output
% . err => error from each presentation (not trial)
% . cosne => cosine from each presentation
% . outacts => the shit it predicted over training
%
% NB: Presentation = individual input/output pair -- trial =
%     sequence of output pairs representing a trial in the seq. 
%     experiment.
%
%     Also: If no criterion desired, use value of numpres for "criterion"
%     
function [wih wch who biash biaso err cosne outacts inacts] = train_srn(sents,targs,hids,numpres,lr,mom,wih,who,wch,biash,biaso)

% get the stuff
%outs = size(targs,2);
%ins = size(sents,2);

% initialize weights from [-.5 to .5]
%  wih = rand(ins,hids) - .5;
%  who = rand(hids,outs) - .5;
%  wch = rand(hids,hids) - .5;
%  biash = zeros(1,hids);
%  biaso = zeros(1,outs);

% initialize context layer to .5's
context = zeros(1,hids) + .5;

i = 1;
hits = [];
err = [];
cosne = [];
outacts = [];
inacts= [];

for epochs=1:numpres,
        
    % take the current sentence, and generate output
    inact = sents(i,:);
    net = inact * wih + context * wch + biash;
    hidact = 1./(1+exp(-net));
    net = hidact * who + biaso;
    outact = 1./(1+exp(-net));

    % retrieve target
    targ = targs(i,:);
    %
    
    %%
    %%
    
    outacts = [outacts ; outact];
    inacts= [inacts; inact];
    err = [err mean((targ - outact).^2)];   
    cosne = [cosne (targ*outact')/(norm(targ)*norm(outact))];
    
    % apply standard backprop
    deltao = outact .* (1 - outact) .* (targ - outact);
    dwho = lr * hidact' * deltao;
    sumterm = deltao * who';
    deltah = hidact .* (1 - hidact) .* sumterm;
    dwih = lr * inact' * deltah;
    dwch = lr * context' * deltah;

    who = who + dwho;
    wch = wch + dwch;
    wih = wih + dwih;

    biash = biash + lr * deltah;
    biaso = biaso + lr * deltao;
    %
    
    context = hidact;
    
    % make sure we don't exceed available in corpus
    i = i + 1;
    if i==size(sents,1),
        i = 1; % if so, let's start over
    end    
    
end

