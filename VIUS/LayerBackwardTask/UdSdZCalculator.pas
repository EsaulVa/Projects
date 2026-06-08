unit UdSdZCalculator;

interface

{$REGION ' uses '}
uses // В АЛФАВИТНОМ ПОРЯДКЕ!
{$I '..\..\..\Common Files\GeneralUses.pas'},
  Classes;
{$ENDREGION ' uses '}

type

  TdSdZCalculator = class(TTunable, IdSdZCalculator)
    public
      function Calculate(const Z, DeltaZ, Percentage: Extended;
        const Surface: IRegularSurface; const ToGR, ToGR1: RVector3;
        var WP: RWindPoint): Extended; virtual; abstract;
  end;

  TdSdZCalculatorI = class(TdSdZCalculator)
    public
      function Calculate(const Z, DeltaZ, Percentage: Extended;
        const Surface: IRegularSurface; const ToGR, ToGR1: RVector3;
        var WP: RWindPoint): Extended; override;
  end;

  TdSdZCalculatorII = class(TdSdZCalculator)
    public
      function Calculate(const Z, DeltaZ, Percentage: Extended;
        const Surface: IRegularSurface; const ToGR, ToGR1: RVector3;
        var WP: RWindPoint): Extended; override;
  end;

implementation

{ TdSdZCalculatorI }

function TdSdZCalculatorI.Calculate(const Z, DeltaZ, Percentage: Extended;
  const Surface: IRegularSurface; const ToGR, ToGR1: RVector3; var WP: RWindPoint): Extended;
const
  Beta = 1;
var
  mv, tau, w: RVector3;
  L, M, N, denominator, Delta, Alfa: Extended;
begin
  try
    with WP do
    begin
      mv := Surface.Normal(U, V, True);
      w := Surface.R(U, V);
      L := Surface.L(U, V);
      M := Surface.M(U, V);
      N := Surface.N(U, V);
    end;
    tau := (ToGR - w).Normalize;
    Delta := tau * mv;
    Alfa := 1 / Sqrt(1 - Sqr(Delta));
    denominator := (ToGR - w).Abs *
      (L * Sqr(wp.dU) + 2 * M * wp.dU * wp.dV + N * sqr(wp.dV));
    Result := (Alfa + Beta * Delta) * ((mv * ToGR1) / denominator);
  except
    Result := INFINITE;
  end;
end;

{ TdSdZCalculatorII }

function TdSdZCalculatorII.Calculate(const Z, DeltaZ, Percentage: Extended;
  const Surface: IRegularSurface; const ToGR, ToGR1: RVector3; var WP: RWindPoint): Extended;
var
  mv, tau, w: RVector3;
  L, M, N, IIKvF: Extended; // Вторая квадратичная форма поверхности
  denominator, Delta, Alfa, Koef, Koef_calc: Extended;
begin
  try
    with WP do
    begin
      mv := Surface.Normal(U, V, True);
      w := Surface.R(U, V);
      L := Surface.L(U, V);
      M := Surface.M(U, V);
      N := Surface.N(U, V);
    end;
    tau := (ToGR - w).Normalize;
    Delta := tau * mv;
    Alfa := 1 / Sqrt(1 - Sqr(Delta));
    IIKvF := -(L * Sqr(wp.dU) + 2 * M * wp.dU * wp.dV + N * Sqr(wp.dV));
    denominator := (ToGR - w).Abs * IIKvF;
    if DeltaZ <> 0 then
      Koef_calc := (-1 / DeltaZ) * Ln(1 - Percentage / 100)
//                        Koef_calc := ( ( -Alfa / obj.Beta)* Mul( Rz, mv ) / znam ) * 1 / ( 1 + ( Alfa / obj.Beta) * ( Delta / IIKvF))
//    if DeltaZ <> 0 then
//      Koef_calc := ((-Alfa / DeltaZ) * (ToGR1 * mv) / denominator) //* 1 / ( 1 - ( Alfa / obj.Beta) * ( Delta / IIKvF))
////      Koef_calc := IIKvF/(Delta*Alfa)-Mul( Rz, mv )/(Absv( Sub( R, w))*Delta)
    else
      Koef_calc := 0;
    Koef := Alfa * Koef_calc * Delta / IIKvF;
//    Koef  := Alfa * obj.Beta * Delta / IIKvF;
    Result := -Alfa * (ToGR1 * mv) / denominator - Koef;
  except
    Result := INFINITE;
  end;
end;

end.
