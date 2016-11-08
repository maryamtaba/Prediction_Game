clear all

% Producing input sentences in the correct format


%sent1='4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1'; % 100%
%sent1='4 2 3 2 1 2 1 2 4 1 3 2 1 4 1 3 1 3 2 4 1 3 1 3 3 4 2 3 4 2 1 3 1 3 4 1 2 1 2 4 2 4 2 1 3 4 2 4'; %random+55%
%sent1='4 2 3 2 1 2 1 2 4 1 3 2 1 4 1 3 1 3 2 4 1 3 1 3 4 1 2 3 4 1 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1'; %random+89%
%sent1='4 2 3 2 1 2 1 2 4 1 3 2 1 4 1 3 1 3 2 4 1 3 1 3 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1'; % random+100%

sentsRand =[];
sentsReg=[];

randsent='4 2 3 2 1 2 1 2 4 1 3 2 1 4 1 3 1 3 2 4 1 3 1 3'; % Regularity = 25%
regularSent='3 4 2 3 4 2 1 3 1 3 4 1 2 1 2 4 2 4 2 1 3 4 2 4'; % Regularity = 55%
%regularSent='4 1 2 3 4 1 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1'; % Regularity = 89%
%regularSent='4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1 4 3 2 1'; % Regularity = 100%


%%Preparing the Random Section of the sequence
sent1= randsent;
% Replacing numbers with corresponding vectors
sent1=strrep(sent1, '1', '0 0 0 1');
sent1=strrep(sent1, '2', '0 0 1 0');
sent1=strrep(sent1, '3', '0 1 0 0');
sent1=strrep(sent1, '4', '1 0 0 0');
sent = str2num(sent1);

% Forming the input matrix
for(i=1:4:length(sent))
    sentsRand = [sentsRand;sent(i:i+3)];
end


%%Preparing the Regular Section of the sequence
sent1= regularSent;
% Replacing numbers with corresponding vectors
sent1=strrep(sent1, '1', '0 0 0 1');
sent1=strrep(sent1, '2', '0 0 1 0');
sent1=strrep(sent1, '3', '0 1 0 0');
sent1=strrep(sent1, '4', '1 0 0 0');
sent = str2num(sent1);

% Forming the input matrix
for(i=1:4:length(sent))
    sentsReg = [sentsReg;sent(i:i+3)];
end


% Initializing parameters
lr = .05;  % Learning rate
mom = .9;  % momentum for escaping local minimums
numpres = 10000; % stopping criterion

% Initializing number of units
hids = 10; % number of hidden units
outs = size(sentsReg,2);
ins = size(sentsReg,2);


%initialize weights from [-.5 to .5]
  wih = rand(ins,hids) - .5;
  who = rand(hids,outs) - .5;
  wch = rand(hids,hids) - .5;
  biash = zeros(1,hids);
  biaso = zeros(1,outs);
  
  
% Producing the output pattern
  targsReg = [sentsReg(2:end,:);sentsReg(1,:)]

% Shuffle the Random section of the input to avoid learning
  sentsRand2 = [];
  for (i = 1:(numpres/24)),
      sentsRand2 = [sentsRand2 ; shuffle(sentsRand,1)];
  end
  sentsRand = sentsRand2;
  targsRand = [sentsRand(2:end,:);sentsRand(1,:)];

  
  % Calling the training function
  [wih wch who biash biaso errRand cosne outacts inacts] = train_srn(sentsRand,targsRand,hids,numpres,lr,mom,wih,who,wch,biash,biaso);
  [wih wch who biash biaso errReg cosne outacts inacts] = train_srn(sentsReg,targsReg,hids,numpres,lr,mom,wih,who,wch,biash,biaso);
  
  plot([errRand errReg])
 % plot(errReg)