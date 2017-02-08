-- parser for the bAbI tasks (based on Matlab code from MemN2N release)
-- returns a datasource object which can then be used to load the data and return minibatches.

local datasource = {}

function datasource:load(data_path, batch_size, memory_size, dict, idict, sentence_length)
    assert(dict ~= nil and idict ~= nil or dict == nil and idict == nil)
    local max_sentence_length = 50
    local max_story_length = 1000
    local max_questions = #data_path * 10000
    local story = torch.zeros(max_questions, max_story_length, max_sentence_length)
    local line_ind, story_ind, sentence_ind, question_ind = 0, 0, 0, 0
    local max_words, max_sentences = 0, 0
    local nwords = (idict ~= nil and #idict) or 0
    local dict = dict or {}
    local idict = idict or {}
    local questions = torch.zeros(max_questions, max_sentence_length)
    local qstory = torch.zeros(max_questions, max_sentence_length)
    local map, is_question

    local fi = 1
    print('reading data: ' .. data_path[fi])
    local fd = io.open(data_path[fi])
    self.blank = ''
    if dict[self.blank] == nil then
        nwords = nwords + 1
        dict[self.blank] = nwords
        idict[nwords] = self.blank
    end
    story:fill(dict[self.blank])
    qstory:fill(dict[self.blank])
    while true do
        local line = fd:read()
        if line == nil then
            fd:close()
            if fi < #data_path then
                fi = fi + 1
                print('reading ' .. data_path[fi])
                fd = io.open(data_path[fi])
                line_ind = 0
                line = fd:read()
            else
                break
            end
        end

        local words = {}
        for w in line:gmatch("%S+") do table.insert(words, w) end
        line_ind = line_ind + 1

        if words[1] == '1' then
            story_ind = story_ind + 1
            sentence_ind = 0
            map = {}
        end

        if string.match(line, '?') == nil then
            is_question = false
            sentence_ind = sentence_ind + 1
        else
           is_question = true
           question_ind = question_ind + 1
           questions[question_ind][1] = story_ind
           questions[question_ind][2] = sentence_ind
        end
        map[#map + 1] = sentence_ind

        for k = 2, #words do
            local w = words[k]
            w = string.lower(w)
            if w:sub(-1) == '.' or w:sub(-1) == '?' then
                w = w:sub(1, -2)
            end
            if dict[w] == nil then
                nwords = nwords + 1
                dict[w] = nwords
                idict[nwords] = w
            end
            max_words = math.max(max_words, k - 1)

            if is_question == false then
                story[story_ind][sentence_ind][k - 1] = dict[w]
            else
               qstory[question_ind][k-1] = dict[w]

                if words[k]:sub(-1) == '?' then
                    answer = words[k+1]
                    answer = string.lower(answer)
                    if dict[answer] == nil then
                        nwords = nwords + 1
                        dict[answer] = nwords
                        idict[nwords] = answer
                    end
                    questions[question_ind][3] = dict[answer]
                    for h = k + 2, #words do
                       questions[question_ind][2+h-k] = map[tonumber(words[h])]
                    end
                    questions[question_ind][10] = line_ind
                    break
                end
            end
        end
        max_sentences = math.max(max_sentences, sentence_ind)
    end
    local sentence_length = sentence_length or max_words
    story = story:sub(1, story_ind, 1, max_sentences, 1, sentence_length):clone()
    questions = questions:sub(1, question_ind):clone()
    qstory = qstory:sub(1, question_ind, 1, sentence_length):clone()

    self.batch_size = batch_size
    if dict == nil then
        -- can change memory size if it's too big
        self.memsize = math.min(memory_size, max_sentences)
        print('memsize: ' .. max_sentences .. ' -> ' .. self.memsize)
        print('train, memsize ' .. self.memsize)
    else
        self.memsize = memory_size
    end
    self.input = torch.zeros(batch_size, story:size(3))
    self.target = torch.zeros(batch_size)
    self.memory = torch.zeros(batch_size, self.memsize, sentence_length)

    self.story = story
    self.qstory = qstory
    self.questions = questions
    self.dict = dict
    self.idict = idict
    self.time = torch.zeros(batch_size, self.memsize)
    for t = 1, self.memsize do
        self.time:select(2, t):fill(t)
    end
    self.shared_entities = torch.zeros(batch_size, self.memsize, self.memsize)

    local locations = {'bedroom', 'kitchen', 'garden', 'hallway', 'bathroom', 'office', 'park', 'cinema'}
    local objects = {'apple', 'football', 'milk', 'table', 'pajamas', 'banana', 'orange', 'pineapple', 'pear', 'melon'}
    local actors = {'john', 'mary', 'sandra', 'daniel', 'jason', 'antoine', 'sumit', 'yann', 'bill', 'fred', 'julie', 'gertrude', 'winona', 'emily', 'jessica', 'lily', 'bernhard', 'greg', 'julius', 'brian'}
    local motivations = {'hungry', 'thirsty', 'bored', 'tired'}
    local containers = {'suitcase', 'container', 'box', 'treasure chest', 'box of chocolates', 'chocolate'}
    local times = {'morning', 'afternoon', 'evening', 'yesterday'}
    local animals = {'mice', 'mouse', 'sheep', 'wolf', 'wolves', 'cat', 'cats', 'frog', 'rhino', 'swan', 'lion'}
    local colors = {'gray', 'yellow', 'green', 'white', 'red', 'blue', 'pink', 'yellow'}
    local shapes = {'square', 'sphere', 'triangle', 'rectangle'}

    local entities = {}
    local function add_to_table(from, to)
        for i = 1, #from do
            table.insert(to, from[i])
        end
    end

    add_to_table(locations, entities)
    add_to_table(objects, entities)
    add_to_table(actors, entities)
    add_to_table(motivations, entities)
    add_to_table(containers, entities)
    add_to_table(times, entities)
    add_to_table(animals, entities)
    add_to_table(colors, entities)
    local entities_indx = {}
    for i = 1, #entities do
        if self.dict[entities[i]] ~= nil then
            table.insert(entities_indx, self.dict[entities[i]])
        end
    end
    self.entities = torch.Tensor(entities_indx)

    self.memory = self.memory:cuda()
    self.input = self.input:cuda()
    self.target = self.target:cuda()
end

function datasource:getBatch(batch)
    self.memory:fill(1)
    self.input:fill(1)
    for b = 1, self.batch_size do
       local d = self.story[self.questions[batch[b]][1]]:sub(1, self.questions[batch[b]][2])
        local offset = math.max(0, d:size(1) - self.memsize)
        d = d:sub(1 + offset, -1)
        self.memory[b]:sub(1, d:size(1), 1, d:size(2)):copy(d)
        self.input[b]:copy(self.qstory[batch[b]])
     end
     self.target:copy(self.questions:index(1, batch:long()):select(2,3))
    return self.input, self.target, self.memory
end


-- convert word indices to string (can be 1D or 2D tensor)
function datasource:text(x)
    local s = ''
    if type(x) == 'number' then
        if self.idict[x] ~= nil then
            s = s .. self.idict[x]
        end
    elseif x:nDimension() == 1 then
        for i = 1, x:size(1) do
            if self.idict[x[i]] ~= nil then
                s = s .. ' ' .. self.idict[x[i]]
            end
        end
    elseif x:nDimension() == 2 then
        for i = 1, x:size(1) do
            if x[i]:ne(self.dict[self.blank]):sum() > 0 then
                for j = 1, x:size(2) do
                    if self.idict[x[i][j]] ~= nil then
                        s = s .. ' ' .. self.idict[x[i][j]]
                    end
                end
                s = s .. '\n'
            end
        end
    end
    return s
end

return datasource
