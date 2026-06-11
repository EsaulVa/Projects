classdef Ellipsoid < Surface
    properties
        a, b, c   % полуоси
    end
    
    methods
        function obj = Ellipsoid(a, b, c)
            obj.a = a;
            obj.b = b;
            obj.c = c;
        end
        
        function r = getPoint(obj, u, v)
            cosu = cos(u); sinu = sin(u);
            cosv = cos(v); sinv = sin(v);
            r = [obj.a * cosu * cosv;
                 obj.b * sinu * cosv;
                 obj.c * sinv];
        end
        
        function [ru, rv] = getFirstDerivatives(obj, u, v)
            cosu = cos(u); sinu = sin(u);
            cosv = cos(v); sinv = sin(v);
            ru = [-obj.a * sinu * cosv;
                   obj.b * cosu * cosv;
                   0];
            rv = [-obj.a * cosu * sinv;
                  -obj.b * sinu * sinv;
                   obj.c * cosv];
        end
        
        function [ruu, ruv, rvv] = getSecondDerivatives(obj, u, v)
            cosu = cos(u); sinu = sin(u);
            cosv = cos(v); sinv = sin(v);
            ruu = [-obj.a * cosu * cosv;
                   -obj.b * sinu * cosv;
                   0];
            ruv = [ obj.a * sinu * sinv;
                   -obj.b * cosu * sinv;
                   0];
            rvv = [-obj.a * cosu * cosv;
                   -obj.b * sinu * cosv;
                   -obj.c * sinv];
        end
    end
end

