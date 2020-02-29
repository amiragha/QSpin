const CRUnion = Union{Float64,Complex128}

struct SpinState{T<:CRUnion}
    dims :: Vector{Int}
    vals :: SparseVector{T}
    # function SpinState(dims::Vector{Int}, vals::SparseVector{T}) where {T<:CRUnion}
    #     prod(dims) == size(vals,1) || error("size for spin state doesn't match", prod(dims), size(vals,1))
    #     new{T}(dims, vals)
    # end
end

# function SpinState(dims::Vector{Int}, vals::SparseVector{T}) where {T<:CRUnion}
#     return SpinState{T}(dims, vals)
# end

@inline nspin(s::SpinState) = length(s.dims)
@inline dimension(s::SpinState) = prod(s.dims)

function kron(s1::SpinState{T}, s2::SpinState{T}) where {T<:CRUnion}
    return SpinState([s1.dims; s2.dims], sparse(kron(s1.vals, s2.vals)))
end

function *(r::R, s::SpinState{T}) where {R<:Number, T<:CRUnion}
    return SpinState(s.dims, r .* s.vals)
end

function +(s1::SpinState{T}, s2::SpinState{T}) where {T<:CRUnion}
    @assert(s1.dims == s2.dims)
    return SpinState(s1.dims, s1.vals + s2.vals)
end

function -(s1::SpinState{T}, s2::SpinState{T}) where {T<:CRUnion}
    @assert(s1.dims == s2.dims)
    return SpinState(s1.dims, s1.vals - s2.vals)
end

function permutespins(s::SpinState{T}, perm::Vector{Int}) where {T<:CRUnion}
    @assert sort(perm) == collect(1:nspin(s))

    nzind = [conf2index(s, index2conf(s, index)[perm]) for index in s.vals.nzind]
    return SpinState(s.dims[perm], sparsevec(nzind, s.vals.nzval, dimension(s)))
end

function index2conf(s::SpinState, index::Int)
    res = zeros(Int, nspin(s))
    idx = index - 1
    for i in reverse(eachindex(s.dims))
        res[i] = idx % s.dims[i]
        idx = div(idx, s.dims[i])
    end
    return res
end

function conf2index(s::SpinState, c::Vector{Int})
    sum(c.* [reverse(cumprod(reverse(s.dims))); 1][2:end]) + 1
end

function confstring(conf::Vector{Int}, dims::Vector{Int};
                    unicode::Bool=true)
    symbs = [
        ["."],
        ["↓", "↑"],
        ["⇓", "⇔", "⇑"],
        ["⤋", "M", "W", "⤊"]
    ]
    res = ""
    for i in eachindex(conf)
        if dims[i] > length(symbs)
            res = res * string(conf[i])
        else
            res = res * symbs[dims[i]][conf[i]+1]
        end
    end
    return res
end

function sshow(s::SpinState; unicode::Bool=true)
    nzind = s.vals.nzind
    for n in eachindex(nzind)
        str = confstring(index2conf(s, nzind[n]), s.dims, unicode=unicode)
        println(str, ":   ", s.vals.nzval[n])
    end
    println()
end
