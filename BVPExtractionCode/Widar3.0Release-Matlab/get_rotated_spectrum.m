% Rotate Velocity Spectrum
% Input:
%       VS: Original Velocity Spectrum sized N*N*T
%       Orientation: Orientation to be rotated
% Output:
%       VS_ro: Rotated Velocity Spectrum
% Author: Yi Zhang
function VS_ro = get_rotated_spectrum(VS, orientation)
    if size(VS,1) ~= size(VS,2)
        error('Not Square Matrix');
    end
    
    N = size(VS,1);
    T = size(VS,3);
    if mod(N,2) == 0
        % Increment to odd row/column
        % Note: velocity_bin = [-1.8:2], hence 
        VS_1 = [zeros(1,N,T); VS(:,:,:)];
        VS_2 = [zeros(N+1,1,T) VS_1(:,:,:)];
        VS_2_ro = imrotate(VS_2, orientation, 'nearest', 'crop');
        VS_ro = VS_2_ro(2:end,2:end,:);
    else
        VS_ro = imrotate(VS, orientation, 'nearest', 'crop');
    end
end