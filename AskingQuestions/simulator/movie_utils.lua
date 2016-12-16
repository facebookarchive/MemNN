-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local pl = require('pl.import_into')()
local tds = require('tds')
local stringx = require('pl.stringx')
local movie_utils={}

function movie_utils:transform(entity)
    local w=pl.utils.split(stringx.strip(entity)," ")
    local string=w[1];
    for i=2,#w do
        string=string.."_"..w[i];
    end
    return string
end

function movie_utils:load_dataset_teacher(fname,CopyAllAnswers)
    local f = io.open(fname)
    local dataset = {}
    while true do
        local s = f:read("*line")
        if s == nil then break end
        if s ~= "" then
            local i1 = s:find(' ')
            local id = s:sub(1, i1 - 1)
            s = s:sub(i1 + 1, -1)
            local w = pl.utils.split(s, '\t')
            local question = w[1]
            local ww
            if w[2]:find(',') then -- if there are multiple answers
                ww = pl.utils.split(w[2], ',')
                -- cleanup the punctuations
                for i = 1, #ww do
                    if ww[i]:sub(1, 1) == ' ' then ww[i] = ww[i]:sub(2, -1) end
                end
                if CopyAllAnswers then
                    ww={ww[1]};
                end
            else
                ww = {w[2]}
            end
            local answer_set = Set(ww)
            for i = 1, answer_set:size() do
                table.insert(dataset, {question,
                                       answer_set:get(i),
                                       answer_set,w[2]})
            end
        end
    end
    return dataset
end

function movie_utils:load_questionEntity(
    fnameData,fnameEntity,frameRelation,framePattern,CopyAllAnswers)
    --each line in fnameEntity contains the entity the question
    --is asking about
    local f = io.open(fnameData)
    local f_entity=io.open(fnameEntity)
    local f_relation=io.open(frameRelation)
    local f_pattern=io.open(framePattern)
    local question_entitys = {}
    local question_relations={}
    local patterns={};
    local categories={};
    while true do
        local s = f:read("*line")
        if s == nil then break end
        local question_entity=stringx.strip(f_entity:read("*line"))
        local question_relation=stringx.strip(f_relation:read("*line"))
        local question_pattern=stringx.strip(f_pattern:read("*line"))
        local question_patterns=pl.utils.split(question_pattern,"\t");
        if s ~= "" then
            local i1 = s:find(' ')
            local _id = s:sub(1, i1 - 1)
            s = s:sub(i1 + 1, -1)
            local w = pl.utils.split(s, '\t')
            local _question= w[1]
            local ww
            if w[2]:find(',') then -- if there are multiple answers
                ww = pl.utils.split(w[2], ',')
                -- cleanup the punctuations
                for i = 1, #ww do
                    if ww[i]:sub(1, 1) == ' ' then ww[i] = ww[i]:sub(2, -1) end
                end
                if CopyAllAnswers then
                    ww={ww[1]};
                end
            else
                ww = {w[2]}
            end
            local answer_set = Set(ww)
            for i = 1, answer_set:size() do
                table.insert(question_entitys,question_entity);
                table.insert(question_relations,question_relation);
                table.insert(patterns,question_patterns[1]);
                table.insert(categories,question_patterns[2]);
            end
        end
    end
    f:close()
    f_entity:close()
    f_relation:close()
    return question_entitys,question_relations,patterns,categories
end

function movie_utils:load_movie_sfs(fname, rels, rels_hint, hints)
    local f = io.open(fname)
    local cnt = 0
    -- hint = {}
    -- rels = {}; rels_hint = {}
    -- rels[1] = 'directed_by'; rels_hint[1] = 'director'
    -- rels[2] = 'written_by'; rels_hint[2] = 'writer'
    -- rels[3] = 'starred_actors'; rels_hint[3] = 'actor'
    -- rels[4] = 'release_year'; rels_hint[4] = 'year'
    -- rels[5] = 'has_genre'; rels_hint[5] = 'genre'
    -- rels[6] = 'has_imdb_votes'; rels_hint[6] = 'popularity'
    -- rels[7] = 'has_tags'; rels_hint[7] = 'tag'
    -- rels[8] = 'in_language'; rels_hint[8] = 'language'
    local kb =  tds.hash()
    local msfs =  tds.hash()

    local function add(e, ind)
        if msfs[e] == nil then
            msfs[e] = torch.Tensor({ind})
        else
            msfs[e] = torch.cat(msfs[e], torch.Tensor({ind}))
        end
    end
    while true do
        local s = f:read("*line")
        if s == nil then break; end
        if s:len() == 0 then s = '1 blah';end
        local i1 = s:find(' ');
        s = s:sub(i1 + 1,-1)
        local s1;
        i1=s:find(" ")
        local e1=s:sub(1,i1-1)
        s1 = s:sub(i1 + 1,-1)
        i1=s1:find(" ")
        local relation=s1:sub(1,i1-1)
        local e2s=s1:sub(i1 + 1,-1)
        local rel_ind = -1
        local i2
        for i = 1, #rels do
            i1, i2 = relation:find(rels[i])
            if i1 ~= nil then rel_ind = i; break; end
        end
        if rel_ind ~= - 1 then
            cnt = cnt + 1
            kb[cnt] = s
            e1=self:transform(e1)
            local w = pl.utils.split(e2s, ', ')
            add(e1, cnt)
            for i = 1, #w do
                local e2 = w[i]
                e2=self:transform(e2)
                if e2:sub(-1)=="?" and rels[rel_ind]=="has_tags" then
                    e2=e2:sub(1,-2)
                end
                if e2:sub(1,1) == ' ' then e2 = e2:sub(2,-1); end
                if e2:sub(-1,-1) == ' ' then e2 = e2:sub(1,-2); end
                add(e2, cnt)
                -- print(e1 .. "<->" .. e2 .. "*")
            end
        end
        if (cnt % 100000) == 0 then
            print(cnt)
            collectgarbage()
            collectgarbage()
        end
    end
    return kb, msfs
end

function movie_utils:load_synonyms(question_template_file)
    local tds = require('tds')
    local f=io.open(question_template_file)
    local category;
    local synonyms={};
    synonyms.str2category=tds.hash()
    synonyms.category2strs=tds.hash()
    synonyms.categoryindex2str=tds.hash()
    while true do
        local s=f:read("*line")
        if s==nil then break;end
        if s:find(":")~=nil then
            local t=s:find(":")
            category=s:sub(t+1,-1);
            synonyms.category2strs[category]=tds.hash();
            synonyms.categoryindex2str[
                #synonyms.categoryindex2str+1]=category;
        elseif s:find("X")~=nil then
            synonyms.str2category[s]=category;
            synonyms.category2strs[category]
                [#synonyms.category2strs[category]+1]=s;
        end
    end
    f:close()
    return synonyms
end

function movie_utils:loadTypo(fname)
    local f=io.open(fname,"r")
    local typo={}
    local typo_reverse={}
    local typo_dev={}
    while true do
        local s=f:read("*line")
        if s==nil then break end
        local split=stringx.split(s," ");
        typo[split[1]]=split[2];
        typo_reverse[#typo_reverse+1]=split[1]
        typo_dev[split[1]]=split[3];
    end
    return typo,typo_reverse,typo_dev
end

function movie_utils:gen_no_list()
    local no_list = {
        "No, that is incorrect.",
        "No, that's wrong.",
        "Wrong.",
        "Sorry, that's not it.",
        "Sorry, wrong.",
        "No."
    }
    return no_list
end


function movie_utils:gen_yes_list()
    local yes_list = {
        "Yes, that is correct!",
        "Yes, that's right.",
        "Correct!",
        "That's right.",
        "That's correct.",
        "Yes!"
    }
    return yes_list
end

function movie_utils:load_movie_answers(fname)
    local f = io.open(fname)
    local cnt = 0
    local answers = {}
    while true do
        local s = f:read("*line")
        if s == nil then break; end
        local w = pl.utils.split(s, '\t')
        answers[#answers+1] = w[1]
        cnt = cnt + 1
    end
    print("loaded " .. #answers .. " answers")
    return answers
end


function movie_utils:load_movie_hints(fname)
    local f = io.open(fname)
    local cnt = 0
    local rels, rels_hint, hint = {}, {}, {}
    rels[1] = 'directed_by'; rels_hint[1] = 'director'
    rels[2] = 'written_by'; rels_hint[2] = 'writer'
    rels[3] = 'starred_actors'; rels_hint[3] = 'actor'
    rels[4] = 'release_year'; rels_hint[4] = 'year'
    rels[5] = 'has_genre'; rels_hint[5] = 'genre'
    rels[6] = 'has_imdb_votes'; rels_hint[6] = 'popularity'
    rels[7] = 'has_tags'; rels_hint[7] = 'tag'
    rels[8] = 'in_language'; rels_hint[8] = 'language'

    while true do
        local s = f:read("*line")
        if s == nil then break; end
        if s:len() == 0 then s = '1 blah';end
        local i1 = s:find(' ');
        s = s:sub(i1+1,-1)
        local rel_ind = -1
        local i2
        for i = 1, #rels do
            i1, i2 = s:find(rels[i])
            if i1 ~=nil then rel_ind = i; break; end
        end
        if rel_ind ~= - 1 then
            local e1 = s:sub(1, i1 -2)
            local e2 = s:sub(i2 + 1, -1)
            hint[e1] = 'movie'
            local w = pl.utils.split(e2, ',')
            for i = 1, #w do
                local e = w[i]
                if e:sub(1,1) == ' ' then e = e:sub(2,-1); end
                if e:sub(-1,-1) == ' ' then e = e:sub(1,-2); end
                if hint[e] == nil then
                    hint[e] = rels_hint[rel_ind]
                else
                    local h = rels_hint[rel_ind]
                    if hint[e]:find(h) == nil then
                        hint[e] = hint[e] .. ', ' .. h
                    end
                end
            end
        end
        cnt = cnt + 1
    end
    print("loaded all hints")
    return rels, rels_hint, hint
end

function movie_utils:load_dataset_student(fname)
    local f = io.open(fname)
    local dataset = {}
    while true do
        local s = f:read("*line")
        if s == nil then break end
        if s ~= "" then
            local i1 = s:find(' ')
            local id = s:sub(1, i1 - 1)
            s = s:sub(i1 + 1, -1)
            local w = pl.utils.split(s, '\t')
            local question = w[1]
            local ww
            if w[2]:find(',') then -- if there are multiple answers
                ww = pl.utils.split(w[2], ',')
                -- cleanup the punctuations
                for i = 1, #ww do
                    if ww[i]:sub(1, 1) == ' ' then ww[i] = ww[i]:sub(2, -1) end
                end
            else
                ww = {w[2]}
            end
            local answer_set = Set(ww)
            if dataset[question] == nil then
                dataset[question] = answer_set
            end
        end
    end
    return dataset
end


function movie_utils:get_supporting_facts(qst, ans, msfs, kb, rels)
    local sfs_hist = {}
    local sfs = {}
    local msf = msfs[ans]
    if msf ~= nil then
        for i = 1, msf:size(1) do
            local k = kb[msf[i]]
            local es = self:extract_entities(k, ans, rels)
            for j = 1, #es do
                if qst:find(es[j]) ~= nil then
                    sfs_hist[#sfs_hist + 1] = k
                    sfs[#sfs + 1] = #sfs + 1
                end
            end
        end
    end
    -- if #sfs > 1 then sumit:enter() end
    return sfs_hist, sfs
end


function movie_utils:extract_entities(str, omit_entity, rels)
    local es = {}
    local function add(e)
        if e ~= omit_entity then
            es[#es + 1] = e
        end
    end
    local rel_ind = -1
    for i = 1, #rels do
        i1, i2 = str:find(rels[i])
        if i1 ~= nil then rel_ind = i; break; end
    end
    if rel_ind ~= - 1 then
        local e1 = str:sub(1, i1 -2)
        add(e1)
        local e2s = str:sub(i2 + 1, -1)
        local w = pl.utils.split(e2s, ',')
        for i = 1, #w do
            e2 = w[i]
            if e2:sub(1,1) == ' ' then e2 = e2:sub(2,-1); end
            if e2:sub(-1,-1) == ' ' then e2 = e2:sub(1,-2); end
            add(e2)
        end
    end
    return es
end

return movie_utils
