local No, parent = torch.class('nn.Normalization', 'nn.Module')


function No:__init()
    parent.__init(self)
    local thresh = thresh or 0
    self.thresh = thresh
end


function No:updateOutput(input)

    self.output:resizeAs(input)
    self.output:copy(input)

    -- store the number of dimensions of the input
    local ldim = input:nDimension()
    -- compute the Euclidean norm over the last dimension of the input
    local norms = input:norm(2,ldim)
    local issmall = norms:le(self.thresh)
    norms[issmall] = (self.thresh+1)
    -- divide the input by the Euclidean norms to produce the output
    self.output:cdiv(norms:expand(self.output:size()))

    return self.output

end


function No:updateGradInput(input,gradOutput)

    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)

    -- store the number of dimensions of the input
    local ldim = input:nDimension()
    -- compute the Euclidean norm over the last dimension of the input
    local norms = input:norm(2,ldim)
    local issmall = norms:le(self.thresh)
    norms[issmall] = (self.thresh+1)
    local proj = self.gradInput

    -- compute the negative of the dot product between the normalized input,
    -- that is, self.output, and gradInput=gradOutput
    local dotprod = proj:clone():cmul(self.output):sum(ldim):mul(-1)
    -- orthogonalize gradInput=gradOutput to the normalized input,
    -- that is, self.output
    proj:add(self.output:clone():cmul(dotprod:expand(proj:size())))
    -- normalize by the norms of the input
    proj:cdiv(norms:expand(proj:size()))

    return proj

end
